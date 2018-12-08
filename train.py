import sys
import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import time

from model import R_pca, time_freq_masking, Model
from datasets import get_dataloader
from utils import reconstruct_wav, get_spec, get_angle, get_mag, save_wav, bss_eval, Scorekeeper, get_batch_spec


scorekeepr = Scorekeeper()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_rpca(dataloader):
	start_time = time.time()
	for batch_idx, (mixed, s1, s2, lengths) in enumerate(dataloader):
		for i in range(len(mixed)):
			mixed_spec = get_spec(mixed[i])
			mixed_mag = get_mag(mixed_spec)
			print(mixed_mag.shape)
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
	model = Model(513, 256).to(device)
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00005)

	for e in range(num_epochs):
		start_time = time.time()
		total_loss = 0.
		for batch_idx, (mixed, s1, s2, lengths) in enumerate(dataloader):
			mixed_spec, s1_spec, s2_spec = get_batch_spec(mixed, s1, s2)
			mixed_mag = torch.Tensor(get_mag(mixed_spec)).to(device)
			mixed_phase = get_angle(mixed_spec)
			s1_mag = torch.Tensor(get_mag(s1_spec)).to(device)
			s2_mag = torch.Tensor(get_mag(s2_spec)).to(device)

			pred_s1, pred_s2 = model(mixed_mag)

			loss = torch.mean((pred_s1-s1_mag)**2 + (pred_s2-s2_mag)**2)
			loss.backward()
			total_loss += loss.item()
			for group in optimizer.param_groups:
				for p in group['params']:
					state = optimizer.state[p]
					if('step' in state and state['step']>=1024):
						state['step'] = 1000
			optimizer.step()

		print(total_loss/len(dataloader))

			# print("pred",pred_s1.shape)
		print(time.time() - start_time)
def main():
	mir1k_data_path = "../data/MIR-1K/Wavfile"

	dataloader = get_dataloader(batch_size=32, shuffle=True, num_workers=0)
	train_rnn(dataloader, 20)	





			

if __name__ == "__main__":
	main()
