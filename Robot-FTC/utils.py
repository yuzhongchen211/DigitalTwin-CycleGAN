import random
import numpy as np
import pickle
import os

class ReplayMemory(object):
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
# if __name__ == '__main__':
#     buffer = ReplayMemory(20000,42)
#     buffer_temp = ReplayMemory(8000,42)
#     buffer_temp.load_buffer('./checkpoints/sac_buffer_ObjectDet_Train_mesh')
#     for i in range(8000):
#         state,obs = buffer_temp.buffer[i]
#         buffer.push(state,obs)
#     for num in ['1','2','3','4','5','6']:
#         buffer_temp = ReplayMemory(2000,42)
#         buffer_temp.load_buffer('./checkpoints/sac_buffer_ObjectDet_Train_mesh_'+num)
#         for i in range(2000):
#             state,obs = buffer_temp.buffer[i]
#             buffer.push(state,obs)
#     buffer.save_buffer('ObjectDet','Train_mesh_20000')
    #     state,reward,done = state
    #     state = state.transpose(0,2,1)
    #     print(state.shape)
    #     break
    #     buffer.buffer[i] = (state,obs)
    # check
    # count = 0
    # for i in range(2000):
    #     state,obs = buffer.buffer[i]
    #     if np.array(state).shape!=(6,180,320):
    #         count += 1
    # print(count)
    # if count == 0:
    #     buffer.save_buffer('ObjectDet', 'Test_pure')

if __name__ == '__main__':
    import json
    buffer = ReplayMemory(2000,42)
    buffer.load_buffer('./checkpoints/sac_buffer_ObjectDet_Test_pure')
    label = {}
    if not os.path.exists('./simu_pure_1k'):
        os,os.mkdir('./simu_pure_1k')
    for i in range(1000):
        state,obs = buffer.buffer[i]
        np.save('./simu_pure_1k/'+str(i)+'.npy',state)
        label[str(i)+'.npy'] = obs
    with open("label_pure_1k.json","w", encoding='utf-8') as f: ## 设置'utf-8'编码
        f.write(json.dumps(label  ,ensure_ascii=False  ,indent=4     )     )