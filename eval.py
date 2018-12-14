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

from model import R_pca, time_freq_masking, BaselineModel, separate_signal_with_RPCA, BaselineModelTemp
# from datasets import get_dataloader
from utils import separate_magnitude_phase, prepare_data_full, wavs_to_specs, save_wav, bss_eval, Scorekeeper, combine_magnitdue_phase, load_wavs, get_specs_transpose



scorekeepr = Scorekeeper()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




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
        print("{}/{}\n".format(step, len(wavs_mono)))
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
    num_hidden_units = 1024
    checkpoint = torch.load("experiment1/0/model_10000.pth")

    mir1k_dir = 'data/MIR1K/MIR-1K'
    test_path = os.path.join(mir1k_dir, 'test_temp.json')
    # test_path = os.path.join(mir1k_dir, 'MIR-1K_test.json')

    with open(test_path, 'r') as text_file:
        content = json.load(text_file)
        # content = text_file.readlines()
    # wav_filenames = [file.strip() for file in content] 
    wav_filenames = ["{}/{}".format("data/MIR1K/MIR-1K/Wavfile", f) for f in content]

    split_size = int(len(wav_filenames)/5.)
    model = BaselineModel(n_fft // 2 + 1, num_hidden_units).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    wavs_src1_pred = list()
    wavs_src2_pred = list()
    model.eval()
    step = 1
    for i in range(5):
        start = i*split_size
        wavs_mono, wavs_src1, wavs_src2 = load_wavs(filenames = wav_filenames[start:start+split_size], sr = mir1k_sr)

        stfts_mono, stfts_src1, stfts_src2 = wavs_to_specs(
            wavs_mono = wavs_mono, wavs_src1 = wavs_src1, wavs_src2 = wavs_src2, n_fft = n_fft, hop_length = hop_length)

        stfts_mono_full, stfts_src1_full, stfts_src2_full = prepare_data_full(stfts_mono = stfts_mono, stfts_src1 = stfts_src1, stfts_src2 = stfts_src2)
        
        with torch.no_grad():
            for wav_filename, wav_mono, stft_mono_full in zip(wav_filenames, wavs_mono, stfts_mono_full):

                stft_mono_magnitude, stft_mono_phase = separate_magnitude_phase(data = stft_mono_full)
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
                print("{}/{}\n".format(step, len(wav_filenames)))
                step += 1
    wavs_mono, wavs_src1, wavs_src2 = load_wavs(filenames = wav_filenames, sr = mir1k_sr)
    gnsdr, gsir, gsar = bss_eval_global(wavs_mono = wavs_mono, wavs_src1 = wavs_src1, wavs_src2 = wavs_src2, wavs_src1_pred = wavs_src1_pred, wavs_src2_pred = wavs_src2_pred)

    print('GNSDR:', gnsdr)
    print('GSIR:', gsir)
    print('GSAR:', gsar)


def main():
    
    eval()      

if __name__ == "__main__":
    main()
