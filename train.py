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

from model import R_pca, time_freq_masking, Model, separate_signal_with_RPCA
from datasets import get_dataloader
from utils import get_spec, get_angle, get_mag, save_wav, bss_eval, Scorekeeper, get_batch_spec, combine_mag_phase, load_wavs, get_specs, get_specs_transpose, wavs_to_specs, sample_data_batch, sperate_magnitude_phase


scorekeepr = Scorekeeper()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_rpca(dataloader):
    start_time = time.time()
    for batch_idx, (mixed, s1, s2, lengths) in enumerate(dataloader):
        for i in range(len(mixed)):
            mixed_spec = get_spec(mixed[i])
            mixed_mag = get_mag(mixed_spec)
            mixed_phase = get_angle(mixed_spec)
            rpca = R_pca(mixed_mag)
            X_music, X_sing = rpca.fit()
            # X_sing, X_music = time_freq_masking(mixed_spec, X_music, X_sing)

            # reconstruct wav
            pred_music_wav = librosa.istft(combine_mag_phase(X_music, mixed_phase))
            pred_sing_wav = librosa.istft(combine_mag_phase(X_sing, mixed_phase))

            nsdr, sir, sar, lens = bss_eval(mixed[i], s1[i], s2[i], pred_music_wav, pred_sing_wav)
            scorekeepr.update(nsdr, sir, sar, lens)
        scorekeepr.print_score()
        
        print("time elasped", time.time() - start_time)
        print("{} / {}".format(batch_idx, len(dataloader)))


def train_rnn():

    mir1k_dir = 'data/MIR1K/MIR-1K'

    # train_path = os.path.join(mir1k_dir, 'train.txt')
    # train_path = os.path.join(mir1k_dir, 'MIR-1K_train.json')
    train_path = os.path.join(mir1k_dir, 'train_temp.json')
    valid_path = os.path.join(mir1k_dir, 'val_temp.json')
    
    wav_filenames_train = []
    # with open(train_path, 'r') as text_file:
    #     content = text_file.readlines()
    # wav_filenames_train = [file.strip() for file in content]
    with open(train_path, 'r') as f:
        content = json.load(f)
    wav_filenames_train = np.array(["{}/{}".format("data/MIR1K/MIR-1K/Wavfile", f) for f in content])
   
    
    with open(valid_path, 'r') as text_file:
        content = json.load(text_file)
    wav_filenames_valid = np.array(["{}/{}".format("data/MIR1K/MIR-1K/Wavfile", f) for f in content])

    # Preprocess parameters
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4
    # Model parameters
    learning_rate = 0.0001
    num_rnn_layer = 3
    num_hidden_units = 256
    batch_size = 64
    sample_frames = 10
    iterations = 100000
    
    train_log_filename = 'train_log_temp.csv'
    clear_tensorboard = False
    model_directory = './model'
    # model_filename = 'svsrnn.ckpt'

    
    # Load train wavs
    # print(len(wavs_mono_train))
    # Turn waves to spectrums
    # print(len(wavs_mono_train))
    split_size = int(len(wav_filenames_train)/4)
    random_wavs = np.random.choice(len(wav_filenames_train), split_size, replace=False)
    wavs_mono_train, wavs_src1_train, wavs_src2_train = load_wavs(filenames = wav_filenames_train[random_wavs], sr = mir1k_sr)

    stfts_mono_train, stfts_src1_train, stfts_src2_train = wavs_to_specs(
    wavs_mono = wavs_mono_train, wavs_src1 = wavs_src1_train, wavs_src2 = wavs_src2_train, n_fft = n_fft, hop_length = hop_length)
            

    
    # Initialize model

    model = Model(n_fft // 2 + 1, num_hidden_units).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)    # 1645314

    losses = []
    step = 1.
    
    # Start training
    start_time = time.time()
    total_loss = 0.

    for i in (range(iterations)):
        model.train()
        data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
            stfts_mono = stfts_mono_train, stfts_src1 = stfts_src1_train, stfts_src2 = stfts_src2_train, batch_size = batch_size, sample_frames = sample_frames)

        x_mixed, _ = sperate_magnitude_phase(data = data_mono_batch)
        y1, _ = sperate_magnitude_phase(data = data_src1_batch)
        y2, _ = sperate_magnitude_phase(data = data_src2_batch)

        optimizer.zero_grad()
        x_mixed = torch.Tensor(x_mixed).to(device)
        y1 = torch.Tensor(y1).to(device)
        y2 = torch.Tensor(y2).to(device)

        pred_s1, pred_s2 = model(x_mixed)
        
        loss = ((y1-pred_s1)**2 + (y2-pred_s2)**2).sum()/y1.data.nelement()

        loss.backward()

        # total_loss += loss.item()
        
        optimizer.step()

        print("iteration: ", i, ", loss: ", loss.item())
        losses.append(loss.item())
        print("time elasped", time.time() - start_time)
        # if i % 2000 == 0:
        #     with open(train_log_filename, 'a') as f:
        #         f.write("{}, {}\n".format(i, loss.item()))
        if i % 2000 == 0:
            # reset 
            wavs_mono_train, wavs_src1_train, wavs_src2_train = None, None, None
            stfts_mono_train, stfts_src1_train, stfts_src2_train = None, None, None
            # start valdiation
            wavs_mono_valid, wavs_src1_valid, wavs_src2_valid = load_wavs(filenames = wav_filenames_valid, sr = mir1k_sr)
            mixed_stft, s1_stft, s2_stft = get_specs_transpose(wavs_mono_valid, wavs_src1_valid, wavs_src2_valid)
            model.eval()
            val_losses = 0.
            with torch.no_grad():
                for j, (mix_spec, s1_spec, s2_spec) in enumerate(zip(mixed_stft, s1_stft, s2_stft)):
                    x_mixed, _ = sperate_magnitude_phase(data = mix_spec)
                    y1, _ = sperate_magnitude_phase(data = s1_spec)
                    y2, _ = sperate_magnitude_phase(data = s2_spec)
                    x_mixed = torch.Tensor(x_mixed).unsqueeze(0).to(device)
                    y1 = torch.Tensor(y1).unsqueeze(0).to(device)
                    y2 = torch.Tensor(y2).unsqueeze(0).to(device)
                    
                    pred_s1, pred_s2 = model(x_mixed)
                    loss = ((y1-pred_s1)**2 + (y2-pred_s2)**2).sum()/y1.data.nelement()
                    val_losses += loss.cpu().numpy()
            with open(train_log_filename, "a") as f:
                f.write("{}, {}, {}\n".format(i, loss.item(), val_losses/len(mixed_stft)))

            if i % 10000==0:
                torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': losses
                }, "model_"+str(i)+".pth")
            wavs_mono_valid, wavs_src1_valid, wavs_src2_valid = None, None, None
            mixed_stft, s1_stft, s2_stft = None, None, None
            random_wavs = np.random.choice(len(wav_filenames_train), split_size, replace=False)
            wavs_mono_train, wavs_src1_train, wavs_src2_train = load_wavs(filenames = wav_filenames_train[random_wavs], sr = mir1k_sr)

            stfts_mono_train, stfts_src1_train, stfts_src2_train = wavs_to_specs(
            wavs_mono = wavs_mono_train, wavs_src1 = wavs_src1_train, wavs_src2 = wavs_src2_train, n_fft = n_fft, hop_length = hop_length)
          
    torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses
        }, "final_model.pth")


if __name__ == "__main__":
    train_rnn()
