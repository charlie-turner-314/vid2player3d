import numpy as np
import torch
import os

print(os.getcwd())

init_condition = torch.load("data/vae/vae_init_condition_djokovic_swing.pt")

print(init_condition)
print(init_condition.shape)

