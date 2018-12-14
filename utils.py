import librosa
import numpy as np
import scipy
import torch
from mir_eval.separation import bss_eval_sources
import os
import json

def load_wav(filename, sr=16000):
    '''
    returns the wav format using sampling rate of 16000
    '''
    data, sr = librosa.load(filename, sr=sr, mono=False)
    mixed = librosa.to_mono(data)*2
    s1, s2 = data[0,:], data[1,:]
    return mixed, s1, s2

def load_wavs(filenames, sr=16000):
    mixed_list = list()
    s1_list = list()
    s2_list = list()
    for filename in filenames:
        mix, s1, s2 = load_wav(filename, sr=sr)
        mixed_list.append(mix)
        s1_list.append(s1)
        s2_list.append(s2)
    return mixed_list, s1_list, s2_list

def separate_magnitude_phase(data):

    return np.abs(data), np.angle(data)

def prepare_data_full(stfts_mono, stfts_src1, stfts_src2):

    stfts_mono_full = list()
    stfts_src1_full = list()
    stfts_src2_full = list()

    for stft_mono, stft_src1, stft_src2 in zip(stfts_mono, stfts_src1, stfts_src2):
        stfts_mono_full.append(stft_mono.transpose())
        stfts_src1_full.append(stft_src1.transpose())
        stfts_src2_full.append(stft_src2.transpose())

    return stfts_mono_full, stfts_src1_full, stfts_src2_full

def combine_magnitdue_phase(magnitudes, phases, hop_length=256):
    '''
    using istft to reconstruct the wav format
    '''
    new_phase = np.exp(1.j * phases)

    return magnitudes*new_phase



def get_spec(wav, n_fft=1024, window="hamming", hop_length=256):
    return librosa.stft(wav, window=window, n_fft=n_fft, hop_length=hop_length)

def get_specs(mixed, s1, s2):
    mixed_stft = list()
    s1_stft = list()
    s2_stft = list()

    
    for mix_, s1_, s2_ in zip(mixed, s1, s2):
        mixed_stft.append(get_spec(mix_))
        s1_stft.append(get_spec(s1_))
        s2_stft.append(get_spec(s2_))

    return mixed_stft, s1_stft, s2_stft

def get_specs_transpose(mixed, s1, s2):
    mixed_stft = list()
    s1_stft = list()
    s2_stft = list()

    for mix_, s1_, s2_ in zip(mixed, s1, s2):
        mixed_stft.append(get_spec(mix_).transpose())
        s1_stft.append(get_spec(s1_).transpose())
        s2_stft.append(get_spec(s2_).transpose())

    return mixed_stft, s1_stft, s2_stft


def save_wav(filename, wav, sr=16000):
    '''
    saves the wav to filename
    '''
    scipy.io.wavfile.write(filename, sr, wav)

def bss_eval(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len = pred_src1_wav.shape[0]
    src1_wav = src1_wav[:len]
    src2_wav = src2_wav[:len]
    mixed_wav = mixed_wav[:len]
    sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                        np.array([pred_src1_wav, pred_src2_wav]), compute_permutation=True)
    sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                          np.array([mixed_wav, mixed_wav]), compute_permutation=True)
    nsdr = sdr - sdr_mixed
    return nsdr, sir, sar, len



def sample_data_batch(stfts_mono, stfts_src1, stfts_src2, batch_size = 64, sample_frames = 8):

    stft_mono_batch = list()
    stft_src1_batch = list()
    stft_src2_batch = list()

    collection_size = len(stfts_mono)
    collection_idx = np.random.choice(collection_size, batch_size, replace = True)
    for idx in collection_idx:
        stft_mono = stfts_mono[idx]
        stft_src1 = stfts_src1[idx]
        stft_src2 = stfts_src2[idx]
        num_frames = stft_mono.shape[1]
        assert  num_frames >= sample_frames
        start = np.random.randint(num_frames - sample_frames + 1)
        end = start + sample_frames
        stft_mono_batch.append(stft_mono[:,start:end])
        stft_src1_batch.append(stft_src1[:,start:end])
        stft_src2_batch.append(stft_src2[:,start:end])

    # Shape: [batch_size, n_frequencies, n_frames]
    stft_mono_batch = np.array(stft_mono_batch)
    stft_src1_batch = np.array(stft_src1_batch)
    stft_src2_batch = np.array(stft_src2_batch)
    # Shape for RNN: [batch_size, n_frames, n_frequencies]
    data_mono_batch = stft_mono_batch.transpose((0, 2, 1))
    data_src1_batch = stft_src1_batch.transpose((0, 2, 1))
    data_src2_batch = stft_src2_batch.transpose((0, 2, 1))

    return data_mono_batch, data_src1_batch, data_src2_batch

def wavs_to_specs(wavs_mono, wavs_src1, wavs_src2, n_fft = 1024, hop_length = 256):
    """
    transform wavs to specs
    """
    stfts_mono = list()
    stfts_src1 = list()
    stfts_src2 = list()

    for wav_mono, wav_src1, wav_src2 in zip(wavs_mono, wavs_src1, wavs_src2):
        # print(wav_mono.shape)
        stft_mono = librosa.stft(wav_mono, n_fft = n_fft, hop_length = hop_length)
        stft_src1 = librosa.stft(wav_src1, n_fft = n_fft, hop_length = hop_length)
        stft_src2 = librosa.stft(wav_src2, n_fft = n_fft, hop_length = hop_length)
        stfts_mono.append(stft_mono)
        stfts_src1.append(stft_src1)
        stfts_src2.append(stft_src2)
        # print(stft_mono.shape)

    return stfts_mono, stfts_src1, stfts_src2

class Scorekeeper():
    def __init__(self):
        self.total_len = 0.
        self.gnsdr = 0.
        self.gsir = 0.
        self.gsar = 0.

    def update(self, nsdr, sir, sar, lens):
        self.total_len += lens
        self.gnsdr += nsdr*lens
        self.gsir += sir*lens
        self.gsar += sar*lens

    def print_score(self):
        print("GNSDR: {},\n GSIR: {},\n GSAR: {}".format
            (self.gnsdr/self.total_len, self.gsir/self.total_len, self.gsar/self.total_len))

def split():
    """
    split 80/10/10 data
    """
    root = "./data/MIR-1K/"
    dataset = "MIR-1K"

    data_path = root + "Wavfile"
    file_list = os.listdir(data_path)


    origin_index = list(range(0, len(file_list)))
    train_ids = np.random.choice(origin_index, 800, replace=False)
    origin_index = [x for x in origin_index if x not in train_ids]
    test_ids = np.random.choice(origin_index, 100, replace=False)
    origin_index = [x for x in origin_index if x not in test_ids]
    val_ids = np.random.choice(origin_index, 100, replace=False)

    train_list = np.array(file_list)[train_ids]
    test_list = np.array(file_list)[test_ids]
    val_list = np.array(file_list)[val_ids]

    train_path = root + dataset + "_train.json"
    test_path = root + dataset + "_test.json"
    val_path = root + dataset + "_val.json"

    print("Saving testing , training ,validation json files...")
    with open (train_path, "w") as f:
        json.dump(train_list.tolist(), f)
    with open (test_path, "w") as f:
        json.dump(test_list.tolist(), f)
    with open (val_path, "w") as f:
        json.dump(val_list.tolist(), f)
def main():
    split()

if __name__ == "__main__":
    main()
        
