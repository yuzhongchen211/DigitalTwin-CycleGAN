

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
import time
import random
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

class Encoder(nn.Module):
    def __init__(self,backbone_name):
        super().__init__()
        backbone = timm.create_model(backbone_name, pretrained=True)
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.embed_dim = self.backbone.num_features
            
    def forward(self,x):
        B,C,H,W = x.shape
        x = x.reshape(2*B,C//2,H,W)
        x = self.backbone(x)
        x = x.reshape(B,2,self.embed_dim)
        x = torch.mean(x,dim=1)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if isinstance(drop, tuple):
            drop_probs = drop
        else:
            drop_probs = (drop, drop)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class DetModel(nn.Module):
    def __init__(self,out_dim,backbone_name='vit_small_patch16'):
        super().__init__()
        self.backbone = Encoder(backbone_name)
        self.head = Mlp(self.backbone.embed_dim,out_features=out_dim)
    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x)
        return x

class Train(object):
    def __init__(self,model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=1e-3,
                                                             total_steps=1000,
                                                             div_factor=25,
                                                             final_div_factor=100,
                                                             three_phase=True)
        self.device = torch.device('cuda:1')
        self.model.to(self.device)

    def learn(self,memory,batch_size):
        self.model.train()
        state_batch, obs_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        pred = self.model(state_batch)
        loss = F.l1_loss(pred, obs_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def test(self,memory,batch_size=256):
        self.model.eval()
        with torch.no_grad():
            state_batch, obs_batch = memory.sample(batch_size=batch_size)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            obs_batch = torch.FloatTensor(obs_batch).to(self.device)
            pred = self.model(state_batch)
            loss = F.l1_loss(10*pred, 10*obs_batch)
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
            'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict()
            }
        torch.save(model,os.path.join('./checkpoints',model_name))
    def load_checkpoint(self,path):
        paraters = torch.load(path,map_location=torch.device('cpu'))
        self.model.load_state_dict(paraters['model'])
# Environment
# env = NormalizedActions(gym.make(args.env_name))
random.seed(6)
URDF_PATH = './ftc_robot/urdf/ftc_robot.urdf'
URDF_OBJECT = './objects/cube_small.urdf'
MAX_STEP = 800
SEVER_ID = p.connect(p.DIRECT)
print('./checkpoints/sac_buffer_ObjectDet_Train_pure_12000')

test_memory = ReplayMemory(12000, 42)
env = World(SEVER_ID, URDF_PATH, URDF_OBJECT, MAX_STEP, p.GUI, True)
ncount = 0 
start_point = time.time()
while ncount < test_memory.capacity:
    env.reset_target()
    for i in range(2):
        env.step([0]*7)
    state,local_bbox,global_bbox = env.robot.capture_image()
    obs = local_bbox+global_bbox
    if obs[0]<0.99 or obs[4]<0.99:
        test_memory.push(state, obs)
        ncount += 1
        print('\r{}|12000'.format(ncount),end='')
print('*****Finish Train Mesh********')
test_memory.save_buffer('ObjectDet','Train_pure_12000')
env.close()
