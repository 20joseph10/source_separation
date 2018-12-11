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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
def demo():
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    num_rnn_layer = 3
    num_hidden_units = 256
    checkpoint = torch.load("final_model.pth")

    mir1k_dir = 'data/MIR1K/MIR-1K'
    test_path = os.path.join(mir1k_dir, 'MIR-1K_test.json')

    with open(test_path, 'r') as text_file:
        content = json.load(text_file)
    wav_filenames = ["{}/{}".format("data/MIR1K/MIR-1K/Wavfile", f) for f in content]
    wav_filenames = ["../HW3/sample_music.wav"] # only get the first two for demo
    
    wavs_mono, wavs_src1, wavs_src2 = load_wavs(filenames = wav_filenames, sr = mir1k_sr)

    stfts_mono, stfts_src1, stfts_src2 = wavs_to_specs(
        wavs_mono = wavs_mono, wavs_src1 = wavs_src1, wavs_src2 = wavs_src2, n_fft = n_fft, hop_length = hop_length)

    stfts_mono_full, stfts_src1_full, stfts_src2_full = prepare_data_full(stfts_mono = stfts_mono, stfts_src1 = stfts_src1, stfts_src2 = stfts_src2)

    model = Model(n_fft // 2 + 1, num_hidden_units).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    wavs_src1_pred = list()
    wavs_src2_pred = list()
    step = 1
    model.eval()
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

            filename = "demo/"+wav_filename.split("/")[-1]
            
            save_wav(filename+"_mono.wav", wav_mono)
            save_wav(filename+"_src1", y1_hat)
            save_wav(filename+"_src2", y2_hat)
    print("done")

def main():
    
    demo()      

if __name__ == "__main__":
    main()
