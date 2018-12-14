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
from torchvision import transforms
from PIL import Image
from matplotlib import cm


from model import R_pca, time_freq_masking, BaselineModel, separate_signal_with_RPCA, BaselineModelTemp
# from datasets import get_dataloader
from utils import save_wav, Scorekeeper, load_wavs, combine_magnitdue_phase, get_specs, get_specs_transpose, wavs_to_specs, sample_data_batch, separate_magnitude_phase

import nni


scorekeepr = Scorekeeper()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_rpca(dataloader):
    start_time = time.time()
    for batch_idx, (mixed, s1, s2, lengths) in enumerate(dataloader):
        for i in range(len(mixed)):
            mixed_spec = get_spec(mixed[i])
            mixed_mag, mixed_phase = separate_magnitude_phase(mixed_spec)
            rpca = R_pca(mixed_mag)
            X_music, X_sing = rpca.fit()
            # X_sing, X_music = time_freq_masking(mixed_spec, X_music, X_sing)

            # reconstruct wav
            pred_music_wav = librosa.istft(combine_magnitdue_phase(X_music, mixed_phase))
            pred_sing_wav = librosa.istft(combine_magnitdue_phase(X_sing, mixed_phase))

            nsdr, sir, sar, lens = bss_eval(mixed[i], s1[i], s2[i], pred_music_wav, pred_sing_wav)
            scorekeepr.update(nsdr, sir, sar, lens)
        scorekeepr.print_score()
        
        print("time elasped", time.time() - start_time)
        print("{} / {}".format(batch_idx, len(dataloader)))

def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if learning_rate <= 1e-5:
        return
    lr = learning_rate * (0.1 ** (epoch // 10000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_rnn(args):

    mir1k_dir = 'data/MIR1K/MIR-1K'
    # train_path = os.path.join(mir1k_dir, 'MIR-1K_train.json')
    # valid_path = os.path.join(mir1k_dir, 'MIR-1K_val.json')

    train_path = os.path.join(mir1k_dir, 'train_temp.json')
    valid_path = os.path.join(mir1k_dir, 'val_temp.json')
    
    wav_filenames_train = []
    
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
    learning_rate = args['learning_rate']
    num_rnn_layer = args['num_layers']
    num_hidden_units = args['hidden_size']
    dropout = args['dropout']
    sample_frames = args['sample_frames']
    save_dirs = "checkpoint"
    batch_size = 64
    iterations = 100000

    # trial_id = nni.get_sequence_id()
    # if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')
    # save_dir = 'checkpoint/trial'+str(trial_id) + "/"
    # os.makedirs(save_dir)
    # # store param in each trail (for testing)
    # with open(save_dir+"params.json", "w") as f:
    #     json.dump(args, f)
    save_dir = "./"
    # train_log_filename = save_dir + 'train_log_temp.csv'
    train_log_filename ='train_log_temp.csv'

    # Load train wavs
    # Turn waves to spectrums
    random_wavs = np.random.choice(len(wav_filenames_train), len(wav_filenames_train), replace=False)
    wavs_mono_train, wavs_src1_train, wavs_src2_train = load_wavs(filenames = wav_filenames_train[random_wavs], sr = mir1k_sr)

    stfts_mono_train, stfts_src1_train, stfts_src2_train = wavs_to_specs(
    wavs_mono = wavs_mono_train, wavs_src1 = wavs_src1_train, wavs_src2 = wavs_src2_train, n_fft = n_fft, hop_length = hop_length)
        
    # Initialize model

    model = BaselineModelTemp(n_fft // 2 , 512, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # 1645314
    loss_fn = nn.MSELoss().to(device)
    step = 1.
    
    start_time = time.time()
    total_loss = 0.
    train_step = 1.
    total_loss = 0.
    best_val_loss = np.inf
    stop = 0
    for i in (range(iterations)):
        model.train()
        data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
            stfts_mono = stfts_mono_train, stfts_src1 = stfts_src1_train, stfts_src2 = stfts_src2_train, batch_size = batch_size, sample_frames = sample_frames)
    
        x_mixed, _ = separate_magnitude_phase(data = data_mono_batch)
        y1, _ = separate_magnitude_phase(data = data_src1_batch)
        y2, _ = separate_magnitude_phase(data = data_src2_batch)

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i, learning_rate)
        
        max_length_even = x_mixed.shape[2]-1 if (x_mixed.shape[2]%2 != 0) else x_mixed.shape[2]

        x_mixed = torch.Tensor(x_mixed[:,:,:max_length_even]).to(device)
        y1 = torch.Tensor(y1[:,:,:max_length_even]).to(device)
        y2 = torch.Tensor(y2[:,:,:max_length_even]).to(device)
        pred_s1, pred_s2 = model(x_mixed)
    
        loss = loss_fn(torch.cat((pred_s1, pred_s2), 1), torch.cat((y1, y2),1))#((y1-pred_s1)**2 + (y2-pred_s2)**2).sum()/y1.data.nelement()
        loss.backward()        
        optimizer.step()
        total_loss += loss.item()

        print("iteration: ", i, ", loss: ", total_loss/train_step)
        print("time elasped", time.time() - start_time)
        
        train_step += 1
        # validate and save progress
        if i % 2000 == 0:
            # reset 
            wavs_mono_train, wavs_src1_train, wavs_src2_train = None, None, None
            stfts_mono_train, stfts_src1_train, stfts_src2_train = None, None, None
            # start valdiation
            wavs_mono_valid, wavs_src1_valid, wavs_src2_valid = load_wavs(filenames = wav_filenames_valid, sr = mir1k_sr)

            mixed_stft, s1_stft, s2_stft = get_specs_transpose(wavs_mono_valid, wavs_src1_valid, wavs_src2_valid)
            val_len = len(mixed_stft)
            model.eval()
            val_losses = 0.
            with torch.no_grad():
                for j, (mix_spec, s1_spec, s2_spec) in enumerate(zip(mixed_stft, s1_stft, s2_stft)):
                    x_mixed, _ = separate_magnitude_phase(data = mix_spec)
                    y1, _ = separate_magnitude_phase(data = s1_spec)
                    y2, _ = separate_magnitude_phase(data = s2_spec)
                    length = x_mixed.shape[0] - x_mixed.shape[0]%2
                    # print(length)
                    x_mixed = torch.Tensor(x_mixed[:length,:512]).unsqueeze(0).to(device)
                    y1 = torch.Tensor(y1[:length,:512]).unsqueeze(0).to(device)
                    y2 = torch.Tensor(y2[:length,:512]).unsqueeze(0).to(device)
                    
                    pred_s1, pred_s2 = model(x_mixed)
                    loss = ((y1-pred_s1)**2 + (y2-pred_s2)**2).sum()/y1.data.nelement()
                    val_losses += loss.cpu().numpy()
            # nni.report_intermediate_result(val_losses/val_len)
            print("{}, {}, {}\n".format(i, total_loss/train_step, val_losses/len(mixed_stft)))

            with open(train_log_filename, "a") as f:
                f.write("{}, {}, {}\n".format(i, total_loss/train_step, val_losses/len(mixed_stft)))
            if best_val_loss > val_losses/(len(mixed_stft)):
                best_val_loss = val_losses/(len(mixed_stft))
                stop = 0
            if best_val_loss < val_losses/(len(mixed_stft)):
                stop += 1
            if stop >= 2 and i >= 10000:
                break
            # if i % 10000==0:
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir+"model_"+str(i)+".pth")
            wavs_mono_valid, wavs_src1_valid, wavs_src2_valid = None, None, None
            mixed_stft, s1_stft, s2_stft = None, None, None
            random_wavs = np.random.choice(len(wav_filenames_train), len(wav_filenames_train), replace=False)
            wavs_mono_train, wavs_src1_train, wavs_src2_train = load_wavs(filenames = wav_filenames_train[random_wavs], sr = mir1k_sr)

            stfts_mono_train, stfts_src1_train, stfts_src2_train = wavs_to_specs(
            wavs_mono = wavs_mono_train, wavs_src1 = wavs_src1_train, wavs_src2 = wavs_src2_train, n_fft = n_fft, hop_length = hop_length)
            train_step = 1.
            total_loss = 0.
    # nni.report_final_result(val_losses/val_len)

    torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_dir+"final_model.pth")

def generate_default_args():
    args = {
        'dropout': 0.25,
        'learning_rate': 0.0001,
        'sample_frames': 64,
        'hidden_size': 256,
        'num_layers': 3,

    }
    return args

if __name__ == "__main__":
    
    try:
        # NEW_ARGS = nni.get_next_parameter()
        args = generate_default_args()
        print(nni.get_sequence_id())
        # args.update(NEW_ARGS)
        print(args)
        train_rnn(args)
    except Exception as exception:
        print(exception)
        raise
