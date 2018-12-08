import librosa
import numpy as np
import scipy
from mir_eval.separation import bss_eval_sources
import os
import json

def load_wav(filename, sr=16000):
	'''
	returns the wav format using sampling rate of 16000
	'''
	data, sr = librosa.load(filename, sr=sr, mono=False)
	mixed = librosa.to_mono(data)
	s1, s2 = data[0,:], data[1,:]
	return mixed, s1, s2

def reconstruct_wav(mag, phase, hop_length=256):
	'''
	using istft to reconstruct the wav format
	'''
	new_phase = np.exp(1.j * phase)
	new_wav = librosa.istft(mag*new_phase, hop_length=hop_length)

	return new_wav

def get_spec(wav, n_fft=1024, window="hamming", hop_length=256):
	return librosa.stft(wav, window=window, n_fft=n_fft, hop_length=hop_length)

def get_angle(spec):
	return np.angle(spec)

def get_mag(spec):
	'''
	should we also take the log?
	'''
	return np.abs(spec)

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
    # sdr, sir, sar, _ = bss_eval_sources(src2_wav,pred_src2_wav, False)
    # sdr_mixed, _, _, _ = bss_eval_sources(src2_wav,mixed_wav, False)
    nsdr = sdr - sdr_mixed
    return nsdr, sir, sar, len

def split():
	root = "data/MIR-1K/"
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

		