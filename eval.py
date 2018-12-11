import sys
import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import time
import librosa
import json
import os

from mir_eval.separation import bss_eval_sources
from model import R_pca, time_freq_masking, Model, separate_signal_with_RPCA
from datasets import get_dataloader
from utils import get_spec, get_angle, get_mag, save_wav, bss_eval, Scorekeeper, get_batch_spec, combine_mag_phase, load_wavs, get_specs_transpose
from preprocess import load_wavs, prepare_data_full, wavs_to_specs, sperate_magnitude_phase, combine_magnitdue_phase


scorekeepr = Scorekeeper()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

def eval_rnn():
    mir1k_data_path = "./data/MIR-1K/Wavfile"

    filenames_json = "./data/MIR1K/MIR-1K/test_temp.json"
    with open(filenames_json, "r") as f:
        filenames_json = json.load(f)
    # print(filenames_json)
    filenames = []
    for (root, dirs, files) in os.walk(mir1k_data_path):
        filenames.extend(['{}/{}'.format(root, f) for f in filenames_json if f.endswith('.wav')])
    
    mixed, s1, s2 = load_wavs(filenames)
    mixed_stft, s1_stft, s2_stft = get_specs_transpose(mixed, s1, s2)
    
    checkpoint = torch.load("model_80000.pth")
    model = Model(513, 256).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])


    pred_s1_list = list()
    pred_s2_list = list()
    with torch.no_grad():
        for i, mix_spec in enumerate(mixed_stft):
            mixed_mag = torch.Tensor([get_mag(mix_spec)]).to(device)
            mixed_phase = get_angle(mix_spec)
            pred_s1, pred_s2 = model(mixed_mag)
            pred_s1 = pred_s1.cpu().numpy()
            pred_s2 = pred_s2.cpu().numpy()
            
            # iterate thru batch
            # for i in range(pred_s1.shape[0]):
            pred_s1_wav = librosa.istft(combine_mag_phase(pred_s1[0], mixed_phase))
            pred_s2_wav = librosa.istft(combine_mag_phase(pred_s2[0], mixed_phase))

            nsdr, sir, sar, lens = bss_eval(mixed[i], s1[i], s2[i], pred_s1_wav, pred_s2_wav)
            scorekeepr.update(nsdr, sir, sar, lens)
            scorekeepr.print_score()
        scorekeepr.print_score()

def bss_eval_global(wavs_mono, wavs_src1, wavs_src2, wavs_src1_pred, wavs_src2_pred):

    assert len(wavs_mono) == len(wavs_src1) == len(wavs_src2) == len(wavs_src1_pred) == len(wavs_src2_pred)

    num_samples = len(wavs_mono)

    gnsdr = np.zeros(2)
    gsir = np.zeros(2)
    gsar = np.zeros(2)
    frames_total = 0
    step = 1
    for wav_mono, wav_src1, wav_src2, wav_src1_pred, wav_src2_pred in zip(wavs_mono, wavs_src1, wavs_src2, wavs_src1_pred, wavs_src2_pred):
        len_cropped = wav_src1_pred.shape[-1]
        wav_mono_cropped = wav_mono[:len_cropped]
        wav_src1_cropped = wav_src1[:len_cropped]
        wav_src2_cropped = wav_src2[:len_cropped]

        sdr, sir, sar, _ = bss_eval_sources(reference_sources = np.asarray([wav_src1_cropped, wav_src2_cropped]), estimated_sources = np.asarray([wav_src1_pred, wav_src2_pred]), compute_permutation = False)
        sdr_mono, _, _, _ = bss_eval_sources(reference_sources = np.asarray([wav_src1_cropped, wav_src2_cropped]), estimated_sources = np.asarray([wav_mono_cropped, wav_mono_cropped]), compute_permutation = False)

        nsdr = sdr - sdr_mono
        gnsdr += len_cropped * nsdr
        gsir += len_cropped * sir
        gsar += len_cropped * sar
        frames_total += len_cropped
        print("{}/{}\n".format(step, 825))
        step += 1

    gnsdr = gnsdr / frames_total
    gsir = gsir / frames_total
    gsar = gsar / frames_total

    return gnsdr, gsir, gsar

def eval():
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    num_rnn_layer = 3
    num_hidden_units = 256
    tensorboard_directory = 'graphs/svsrnn'
    clear_tensorboard = False
    model_directory = 'model'
    checkpoint = torch.load("final_model.pth")

    mir1k_dir = 'data/MIR1K/MIR-1K'
    test_path = os.path.join(mir1k_dir, 'test_temp.json')
    #test_path = os.path.join(mir1k_dir, 'MIR-1K_test.json')

    with open(test_path, 'r') as text_file:
        content = json.load(text_file)
        # content = text_file.readlines()
    # wav_filenames = [file.strip() for file in content] 
    wav_filenames = ["{}/{}".format("data/MIR1K/MIR-1K/Wavfile", f) for f in content]
    print(wav_filenames)
    print(len(wav_filenames))

    #wav_filenames = ['small_test_data/yifen_4_10.wav', 'small_test_data/yifen_5_10.wav']
    # output_directory = 'demo'
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)

    wavs_mono, wavs_src1, wavs_src2 = load_wavs(filenames = wav_filenames, sr = mir1k_sr)

    stfts_mono, stfts_src1, stfts_src2 = wavs_to_specs(
        wavs_mono = wavs_mono, wavs_src1 = wavs_src1, wavs_src2 = wavs_src2, n_fft = n_fft, hop_length = hop_length)

    stfts_mono_full, stfts_src1_full, stfts_src2_full = prepare_data_full(stfts_mono = stfts_mono, stfts_src1 = stfts_src1, stfts_src2 = stfts_src2)

    model = Model(n_fft // 2 + 1, num_hidden_units).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    wavs_src1_pred = list()
    wavs_src2_pred = list()
    step = 1
    with torch.no_grad():
        for wav_filename, wav_mono, stft_mono_full in zip(wav_filenames, wavs_mono, stfts_mono_full):

            stft_mono_magnitude, stft_mono_phase = sperate_magnitude_phase(data = stft_mono_full)
            stft_mono_magnitude = np.array([stft_mono_magnitude])

            stft_mono_magnitude = torch.Tensor(stft_mono_magnitude).to(device)
            y1_pred, y2_pred = model(stft_mono_magnitude)

            # ISTFT with the phase from mono
            y1_pred = y1_pred.cpu().numpy()
            y2_pred = y2_pred.cpu().numpy()
            y1_stft_hat = combine_magnitdue_phase(magnitudes = y1_pred[0], phases = stft_mono_phase)
            y2_stft_hat = combine_magnitdue_phase(magnitudes = y2_pred[0], phases = stft_mono_phase)

            y1_stft_hat = y1_stft_hat.transpose()
            y2_stft_hat = y2_stft_hat.transpose()

            y1_hat = librosa.istft(y1_stft_hat, hop_length = hop_length)
            y2_hat = librosa.istft(y2_stft_hat, hop_length = hop_length)

            wavs_src1_pred.append(y1_hat)
            wavs_src2_pred.append(y2_hat)
            print("{}/{}\n".format(step, 825))
            step += 1

    gnsdr, gsir, gsar = bss_eval_global(wavs_mono = wavs_mono, wavs_src1 = wavs_src1, wavs_src2 = wavs_src2, wavs_src1_pred = wavs_src1_pred, wavs_src2_pred = wavs_src2_pred)

    print('GNSDR:', gnsdr)
    print('GSIR:', gsir)
    print('GSAR:', gsar)


def main():
    
    eval()      

if __name__ == "__main__":
    main()
