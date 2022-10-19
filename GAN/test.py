#!/usr/bin/python3

import argparse
import sys
import os
import random
import numpy as np
import pickle
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./Data/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=6, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=6, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_cycle_A2B_withdet_60.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_cycle_B2A_withdet_60.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
opt.size = (180, 320)
path_name = './output/cycle'
print(opt)
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size[0], opt.size[1])
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size[0], opt.size[1])

# Dataset loader
transforms_ = [ transforms.ToTensor() ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='train'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists(path_name):
    os.mkdir(path_name)
for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)
    # Save image files
    save_image(fake_A[0,:3,:,:], path_name+'/real_local_%04d.png' % (i+1))
    save_image(fake_A[0,3:,:,:], path_name+'/real_global_%04d.png' % (i+1))
    save_image(fake_B[0,:3,:,:], path_name+'/sim_local_%04d.png' % (i+1))
    save_image(fake_B[0,3:,:,:], path_name+'/sim_global_%04d.png' % (i+1))
    save_image(real_A[0,:3,:,:], path_name+'/real_local_raw_%04d.png' % (i+1))
    save_image(real_A[0,3:,:,:], path_name+'/real_global_raw_%04d.png' % (i+1))
    save_image(real_B[0,:3,:,:], path_name+'/sim_local_raw_%04d.png' % (i+1))
    save_image(real_B[0,3:,:,:], path_name+'/sim_global_raw_%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
    if i>=50: break

sys.stdout.write('\n')
###################################
class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, obs):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, obs)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, obs = map(np.stack, zip(*batch))
        return state, obs

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity