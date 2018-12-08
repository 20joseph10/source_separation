import sys
import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import time

from model import R_pca, time_freq_masking, Model
from datasets import get_dataloader
from utils import reconstruct_wav, get_spec, get_angle, get_mag, save_wav, bss_eval, Scorekeeper


scorekeepr = Scorekeeper()


def train_rpca(dataloader):
	start_time = time.time()
	for batch_idx, (mixed, s1, s2, lengths) in enumerate(dataloader):
		for i in range(len(mixed)):
			mixed_spec = get_spec(mixed[i])
			mixed_mag = get_mag(mixed_spec)
			mixed_phase = get_angle(mixed_spec)
			rpca = R_pca(mixed_mag)
			X_music, X_sing = rpca.fit()
			X_sing, X_music = time_freq_masking(mixed_mag, X_music, X_sing, gain=1)

			# reconstruct wav
			pred_music_wav = reconstruct_wav(X_music, mixed_phase)
			pred_sing_wav = reconstruct_wav(X_sing, mixed_phase)

			nsdr, sir, sar, lens = bss_eval(mixed[i], s1[i], s2[i], pred_music_wav, pred_sing_wav)
			
		scorekeepr.update(nsdr, sir, sar, lens)
		scorekeepr.print_score()
		
		print("time elasped", time.time() - start_time)
		print("{} / {}".format(batch_idx, len(dataloader)))

def train_rnn(dataloader, num_epochs):
	# model = Model()
	for batch_idx, (mixed, s1, s2, lengths) in enumerate(dataloader):
		for i in range(len(mixed)):
			mixed_spec = get_spec(mixed[i])
			mixed_mag = get_mag(mixed_spec)
			print(mixed_mag.shape)
			break
		break

def main():
	mir1k_data_path = "../data/MIR-1K/Wavfile"

	dataloader = get_dataloader(batch_size=4, shuffle=True, num_workers=0)
	train_rnn(dataloader, 5)
	





			

if __name__ == "__main__":
	main()
