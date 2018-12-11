import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import load_wav
from torchvision import transforms
from utils import get_spec, get_angle, get_mag, save_wav, bss_eval
import json


class MIR1K(Dataset):

	def __init__(self, root, mode="train", sr=16000, transform=None):
		self.root = root
		self.transform = transform
		self.wavfiles = []
		self.sr = 16000
		self.filenames_json = "./data/MIR-1K/MIR-1K_"+mode+".json"
		with open(self.filenames_json, "r") as f:
			self.filenames = json.load(f)
		for (root, dirs, files) in os.walk(root):
			self.wavfiles.extend(['{}/{}'.format(root, f) for f in self.filenames if f.endswith('.wav')])

	def __getitem__(self, index):
		'''
		returns the file in wav format
		'''
		wavfile = self.wavfiles[index]

		mixed, s1, s2 = load_wav(wavfile, sr=self.sr)
		
		return mixed, s1, s2, len(mixed)

	def __len__(self):
		return len(self.wavfiles)


def collate_fn(data):
	# data.sort(key=lambda x: len(x[1]), reverse=True)
	mixed, s1, s2, lengths = zip(*data)
	max_length = np.max(lengths)

	new_mixed = np.zeros((len(mixed), max_length))
	new_s1 = np.zeros((len(s1), max_length))
	new_s2 = np.zeros((len(s2), max_length))
	# padd the wav to same size
	for i in range(len(mixed)):
		new_mixed[i,:lengths[i]] = mixed[i]
		new_s1[i,:lengths[i]] = s1[i]
		new_s2[i,:lengths[i]] = s2[i]

	return new_mixed, new_s1, new_s2, lengths


def get_dataloader(mode="train", batch_size=4, shuffle=True, num_workers=0):
	mir1k_data_path = "./data/MIR-1K/Wavfile"
	dataset = MIR1K(root=mir1k_data_path, mode=mode)
	dataloader = DataLoader(dataset=dataset,
							batch_size=batch_size,
							shuffle=shuffle,
							num_workers=num_workers,
							collate_fn=collate_fn
							)
	return dataloader

def main():
	mir1k_data_path = "./data/MIR-1K/Wavfile"
	dataset = MIR1K(root=mir1k_data_path, mode="test")
	print(len(dataset))
	# dataloader = DataLoader(dataset=dataset,
	# 						batch_size=4,
	# 						shuffle=True,
	# 						num_workers=0,
	# 						collate_fn=collate_fn
	# 						)
	# import time
	# start_time = time.time()
	# for batch_idx, (mixed, s1, s2, lengths) in enumerate(dataloader):
	# 	print("{}/{}".format(batch_idx, len(dataloader)))

	# print(time.time()-start_time)
	
	
if __name__ == "__main__":
	main()