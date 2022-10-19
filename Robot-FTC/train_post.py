

import pybullet as p

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from Env.world import World
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import random
import os
import pickle
from tqdm import tqdm
from model import DetModel
from torchvision import transforms
import numpy as np
_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 缩放
    transforms.Normalize([0.5]*6, [0.5]*6),  # 标准化均值为0标准差为1
])

#%%
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


class Train(object):
    def __init__(self,model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=1e-3,
                                                             total_steps=500,
                                                             div_factor=25,
                                                             final_div_factor=100,
                                                             three_phase=True)
        self.device = torch.device('cuda:3')
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.model.to(self.device)

    def learn(self,memory,batch_size):
        self.model.train()
        state_batch, obs_batch = memory.sample(batch_size=batch_size)
        state_batch = _transform(torch.FloatTensor(state_batch).to(self.device))
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        pred = self.model(state_batch)
        loss = self.loss_fn(10*pred, 10*obs_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def test(self,memory,batch_size=256):
        self.model.eval()
        with torch.no_grad():
            state_batch, obs_batch = memory.sample(batch_size=batch_size)
            state_batch = _transform(torch.FloatTensor(state_batch).to(self.device))
            obs_batch = torch.FloatTensor(obs_batch).to(self.device)
            pred = self.model(state_batch)
            loss = self.loss_fn(10*pred, 10*obs_batch)
        return loss.item()

    def sample(self,state):
        state = torch.unsqueeze(torch.FloatTensor(state),dim=0).to(self.device)
        self.model.eval()
        obs = self.model(state)
        obs = obs.detach().cpu().numpy()[0]
        self.model.train()
        return obs
        
    def save_checkpoint(self,model_name='last_model'):
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        model = {
            'model':self.model.state_dict()
            }
        torch.save(model,os.path.join('./checkpoints',model_name))
    def load_checkpoint(self,path):
        paraters = torch.load(path,map_location=torch.device('cpu'))
        self.model.load_state_dict(paraters['model'])
# Environment
# env = NormalizedActions(gym.make(args.env_name))

URDF_PATH = './ftc_robot/urdf/ftc_robot.urdf'
URDF_OBJECT = './objects/cube_small.urdf'
MAX_STEP = 800
SEVER_ID = p.connect(p.DIRECT)

if os.path.exists('./checkpoints/sac_buffer_CycleGAN_60'):
    train_memory = ReplayMemory(8000, 42)
    train_memory.load_buffer('./checkpoints/sac_buffer_CycleGAN_60')
# else:
#     train_memory = ReplayMemory(8000, 42)
#     env = World(SEVER_ID, URDF_PATH, URDF_OBJECT, MAX_STEP, p.GUI, False)
#     for i in tqdm(range(train_memory.capacity)):
#         state = env.reset()
#         for i in range(20):
#             state = env.step([0]*7)
#         obs, _ = p.getBasePositionAndOrientation(env.object)
#         train_memory.push(state, obs)
#     train_memory.save_buffer('ObjectDet', 'Train_mesh')
#     env.close()

agent = Train(model=DetModel(4,'swin_tiny_patch4_window7_224'))
agent.load_checkpoint('./checkpoints/ObjectDet_Swin_T_mesh')
writer = SummaryWriter('runs/Object_detect_pure_swin-ti-cycleagan')

# Training loop
updates = 0
total_numsteps = 0
batch_size = 32
error_test = np.zeros(50)
for i_episode in range(500):
    train_loss = agent.learn(train_memory,batch_size)
    writer.add_scalar('Loss/Train', train_loss,i_episode)
    error_test[i_episode%50] = train_loss
    print('{}|{}  Train error:{:.4f}'.format(i_episode,500,train_loss))
agent.save_checkpoint('CycleGAN_20.pth')
print('Error: {:.4f} + {:.6f}'.format(np.mean(error_test),np.std(error_test)))
