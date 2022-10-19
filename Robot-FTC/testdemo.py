from Env.world import World
import pybullet as p
import torch
from model import DetModel
from torchvision import transforms
import os 

_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 缩放
    transforms.Normalize([0.5]*6, [0.5]*6),  # 标准化均值为0标准差为1
])
class Train(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             max_lr=1e-3,
                                                             total_steps=2000,
                                                             div_factor=25,
                                                             final_div_factor=100,
                                                             three_phase=True)
        self.device = torch.device('cpu')
        self.model.to(self.device)

    def learn(self, memory, batch_size):
        self.model.train()
        state_batch, obs_batch = memory.sample(batch_size=batch_size)
        state_batch = _transform(torch.FloatTensor(state_batch).to(self.device))
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        pred = self.model(state_batch)
        loss = F.l1_loss(pred, obs_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, memory, batch_size=256):
        self.model.eval()
        with torch.no_grad():
            state_batch, obs_batch = memory.sample(batch_size=batch_size)
            state_batch = _transform(torch.FloatTensor(state_batch).to(self.device))
            obs_batch = torch.FloatTensor(obs_batch).to(self.device)
            pred = self.model(state_batch)
            loss = F.l1_loss(10 * pred, 10 * obs_batch)
        return loss.item()

    def sample(self, state):
        state = _transform(torch.unsqueeze(torch.FloatTensor(state), dim=0)).to(self.device)
        self.model.eval()
        obs = self.model(state)
        obs = obs.detach().cpu().numpy()[0]
        self.model.train()
        return obs

    def save_checkpoint(self, model_name='last_model'):
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        model = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(model, os.path.join('./checkpoints', model_name))

    def load_checkpoint(self, path):
        paraters = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(paraters['model'])
# class Move(object):
#     def __init__(self,env) -> None:
#         self.env = env
#         self.arm_down = False
#         self.world_target_global = np.array([0.8680, 0.6415])
#         self.world_target_local = np.array([0.5877,0.6937])
#         self.world_target_local_down = np.array([0.5606,0.5013])
#         self.pid_0 = PID(0.1, 0.2, 0.03, setpoint=0)
#         self.pid_1 = PID(0.5, 0.1, 0.05, setpoint=0)
#     def get_action_from_obs(self,obs_target):
#         if abs(obs_target[2]-obs_target[0])<0.7:
#             obs_pos = np.array([obs_target[0]/2+obs_target[2]/2,obs_target[1]/2+obs_target[3]/2])
#             if not self.arm_down:
#                 error_1 = obs_pos - self.world_target_local
#             else:
#                 error_1 = obs_pos - self.world_target_local_down
#         else:
#             error_1 = np.zeros(2)
#         if abs(obs_target[6]-obs_target[4])<0.7:
#             obs_pos = np.array([obs_target[4]/2+obs_target[6]/2,obs_target[5]/2+obs_target[7]/2])
#             error_2 = obs_pos - self.world_target_global   
#         else:
#             error_2 = np.zeros(2)
#         if self.arm_down:
#             error = 0.8*error_1 + 0.2*error_2
#         else:
#             error = 0.2*error_1 + 0.8*error_2
#         dist = (error[0] ** 2 + error[1] ** 2) ** 0.5
#         print('\r Step:%3d Dist: %.4f   Error:%.4f  %.4f'%(self.env.step_num,dist,error[0],error[1]),end='')
#         if dist>=0.008 and dist <=0.15:# 误差在0.04以内都可抓取，这里设置为0.015
#             if not self.arm_down:
#                 self.env.down_gripper()
#                 self.arm_down = True
#             control_v = self.pid_0(error[0])
#             rgb_state,reward, done = self.env.step([control_v, control_v, control_v, control_v, 0, 0, 0])
#             control_v = self.pid_1(error[1])
#             rgb_state, reward, done = self.env.step([-control_v,-control_v,control_v,control_v,0,0,0])
#             return rgb_state, reward, done
#         elif dist>0.15:# 误差在0.04以内都可抓取，这里设置为0.015
#             if dist > 0.25: # 不同视角误差不同，避免降下爪子误差增大，再抬上去的情况
#                 if self.arm_down:
#                     self.env.up_gripper()
#                     self.arm_down = False
#             control_v = self.pid_0(error[0])
#             rgb_state,reward, done = self.env.step([control_v, control_v, control_v, control_v, 0, 0, 0])
#             control_v = self.pid_1(error[1])
#             rgb_state, reward, done = self.env.step([-control_v,-control_v,control_v,control_v,0,0,0])
#             return rgb_state, reward, done
#         else:
#             if not self.env.done:
#                 rgb_state, reward, done = self.env.start_gripper()
#             else:
#                 rgb_state, reward, done = self.env.step([0, 0, 0, 0, 0, -5, 5])
#             return rgb_state, reward, done

URDF_PATH = './ftc_robot/urdf/ftc_robot.urdf'
URDF_OBJECT = './objects/cube_small.urdf'
MAX_STEP = 1200
SEVER_ID = p.connect(p.DIRECT)
world_target_position = [0.6316236973256792, -0.1460444541934986, 0.007560317191009275]
base_position = [-0.015428701779833326, -0.05041386853709302, 0.26249300868157094]
base_target_position = [0.6458283388116746, -0.09560831301991306, -0.25495839295994416]
world_target_global = [0.8680, 0.6415]
world_target_local = [0.5877,0.6937]
world_target_local_down = [0.5606,0.5013]
import cv2
import numpy as np
def get_camera(state, obs_position, server_id, width=640, height=360, step=None):
        camera_pos = [2,-1.0,1.5]
        camera_target_pos = [0.80,0,0]
        camera_up_vector = [0,0,1]
        view_mat = p.computeViewMatrix(camera_pos, camera_target_pos, camera_up_vector, server_id)
        proj_mat = p.computeProjectionMatrixFOV(fov=49.1,
												aspect=1.0,
												nearVal=0.1,
												farVal=100,
												physicsClientId=server_id)
        w, h, rgb, depth, seg = p.getCameraImage(width=width,
												 height=height,
												 viewMatrix=view_mat,
												 projectionMatrix=proj_mat,
												 renderer=p.ER_BULLET_HARDWARE_OPENGL)


        rgb = np.array(rgb,dtype=np.uint8)[:,:,:3][:,:,::-1].copy()
        cv2.imwrite('ones/%s_all.png'%step, rgb)


        state = 255 * state

        state_global = state.astype(np.uint8)[3:,:,:].transpose(1,2,0)[:,:,::-1].copy()
        start_point,end_point = (int(obs_position[4]*320),int(obs_position[5]*180)),(int(obs_position[6]*320),int(obs_position[7]*180))
        color_image_global = cv2.rectangle(state_global,start_point,end_point,(0,255,0),2)

        state_local = state.astype(np.uint8)[:3,:,:].transpose(1,2,0)[:,:,::-1].copy()
        start_point,end_point = (int(obs_position[0]*320),int(obs_position[1]*180)),(int(obs_position[2]*320),int(obs_position[3]*180))
        color_image_local = cv2.rectangle(state_local,start_point,end_point,(0,255,0),2)

        cv2.imwrite('global/%d_global.png'%step, color_image_global)   
        cv2.imwrite('local/%s_lcoal.png'%step, color_image_local)
if __name__ == '__main__':
    env = World(SEVER_ID, URDF_PATH, URDF_OBJECT, MAX_STEP, p.DIRECT, True)
    agent = Train(model=DetModel(4, 'swin_tiny_patch4_window7_224'))
    model_path = './checkpoints/ObjectDet_Swin_Ti_14k'
    agent.load_checkpoint(model_path)
    success_rate = 0
    x0,y0=[],[]
    a0,b0=[],[]
    for i in range(20):
        state = env.reset()
        # os.mkdir(env.root)
        # os.mkdir(env.root+'/global')
        # os.mkdir(env.root+'/local')
        # os.mkdir(env.root+'/ones')
        while not env.done:
            state,local_bbox,global_bbox = env.robot.capture_image()
            obs = agent.sample(state)
            env.get_action_from_obs(obs)
            # env.step([0]*7)
            # #print(local_bbox,global_bbox)
            # if env.step_num > 3:
            #     x0.append((local_bbox[0]+local_bbox[2])/2)
            #     y0.append((local_bbox[1]+local_bbox[3])/2)
            #     a0.append((global_bbox[0]+global_bbox[2])/2)
            #     b0.append((global_bbox[1]+global_bbox[3])/2)
            #     print(a0,b0)
            #     print(sum(x0)/len(x0),sum(y0)/len(y0),sum(a0)/len(a0),sum(b0)/len(b0))
            #     #  env.start_gripper()
            #     break
            # obs = agent.sample(state)
            # state,_,_ = env.get_action_from_obs(obs)
            #state = 255 * state
            # state = state.astype(np.uint8)[3:,:,:]
            # state = state.transpose(2,1,0)
            # print(state.shape)
            # # color_images = cv2.rectangle(color_images,(round(box1[0]), round(box1[1])),(round(box1[2]), round(box1[3])),(0,255,0),2)
            # start_point,end_point = (int(obs_position[4]*180),int(obs_position[5]*320)),(int(obs_position[6]*180),int(obs_position[7]*320))
            # color_image = cv2.rectangle(state,start_point,end_point,(0,255,0),2)

            # cv2.imwrite('temp/%d_global.png'%i, color_image)   
            if env.done and env.step_num<env.max_step:
                success_rate +=1
        print('Success Rate: {:.2f} \n {}'.format(success_rate / (i+1), model_path))
    env.close()




