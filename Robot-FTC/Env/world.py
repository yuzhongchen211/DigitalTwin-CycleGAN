import os

import pybullet as p
import pybullet_data
import random
from .robot import Robot
from glob import glob
import math
import numpy as np
import cv2
import time

world_target_position = [0.6316236973256792, -0.1460444541934986, 0.007560317191009275]
base_position = [-0.015428701779833326, -0.05041386853709302, 0.26249300868157094]
base_target_position = [0.6458283388116746, -0.09560831301991306, -0.25495839295994416]
world_target_global = [0.8680, 0.6415]
world_target_local = [0.5877,0.6937]
world_target_local_down = [0.5606,0.5013]
class World(object):
	def __init__(self, server_id, urdf_path, urdf_object, max_step=1000, mode=p.GUI, pure=True):
		self.server_id = server_id
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
		p.setRealTimeSimulation(0)
		p.setGravity(0, 0, -9.8)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self.robot = Robot(robot_urdf=urdf_path, server_id=self.server_id)
		self.plane_id = p.loadURDF('plane.urdf')
		self.pure = pure
		if not self.pure:
			texture_list = glob('./Mesh/swirly/*.jpg')
			texture_path = random.choice(texture_list)
			textureId = p.loadTexture(texture_path)
			p.changeVisualShape(self.plane_id, -1, textureUniqueId=textureId)
		self.urdf_path = urdf_path
		self.urdf_object = urdf_object
		self.max_step = max_step
		self.mode = mode
		self.step_num = 0
		self.done = False
		pos = [0.7 + 0.5 * random.random(), -0.3 + 0.6 * random.random(), 0.018]
		#pos = [world_target_position[0],world_target_position[1],0.018]
		ang = -3.14 * 0.5 + 3.1415925438 * random.random()
		orn = p.getQuaternionFromEuler([0, 0, ang])
		self.object = p.loadURDF(self.urdf_object, pos, orn)
		self.robot.reset_arm(False)
		self.error_1_old = np.array([0,0])
		self.error_2_old = np.array([0,0])
		self.root = '%d'%time.time()

	def reset_target(self):
		self.step_num = 0
		self.done = False
		p.removeBody(self.object)
		if random.random()>2:
			pos = [0.7 + 0.7 * random.random(), -0.4 + 0.8 * random.random(), 0.018]
			self.robot.reset_arm(False)
		else:
			pos = [world_target_position[0]+0.15*random.random(), world_target_position[1]+0.018*random.random()-0.009, 0.018]
			self.robot.reset_arm(True)
		pos = [world_target_position[0],world_target_position[1],0.018]
		ang = -3.14 * 0.5 + 3.1415925438 * random.random()
		orn = p.getQuaternionFromEuler([0, 0, ang])
		self.object = p.loadURDF(self.urdf_object, pos, orn)
		p.removeBody(self.plane_id)
		self.plane_id = p.loadURDF('plane.urdf')
		if not self.pure:
			texture_list = glob('./Mesh/swirly/*.jpg')
			texture_path = random.choice(texture_list)
			textureId = p.loadTexture(texture_path)
			p.changeVisualShape(self.plane_id, -1, textureUniqueId=textureId)

	def step(self, action_index,return_with_bbox=False):
		rgb_state,local_bbox,global_bbox = self.robot.step(action_index)
		# print(global_bbox)
		# self.get_camera(self.root,rgb_state,step=self.step_num)
		self.step_num += 1
		reward = self.reward_func()
		if self.step_num > self.max_step:
			self.done = True
		if return_with_bbox:
			return rgb_state, local_bbox,global_bbox,reward, self.done
		else:
			return rgb_state, reward, self.done

	def reward_func(self):
		return 0.0

	def _GetClockAngle(self, v1, v2):
		# 2个向量模的乘积
		TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
		# 叉乘
		rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / TheNorm))
		# 点乘
		theta = np.arccos(np.dot(v1, v2) / TheNorm)
		if rho < 0:
			return -theta
		else:
			return theta

	def get_camera(self, root,state, width=640, height=360, step=None):
		camera_pos = [2,-1.0,1.5]
		camera_target_pos = [0.80,0,0]
		camera_up_vector = [0,0,1]
		view_mat = p.computeViewMatrix(camera_pos, camera_target_pos, camera_up_vector, self.server_id)
		proj_mat = p.computeProjectionMatrixFOV(fov=49.1,
												aspect=1.0,
												nearVal=0.1,
												farVal=100,
												physicsClientId=self.server_id)
		w, h, rgb, depth, seg = p.getCameraImage(width=width,
												 height=height,
												 viewMatrix=view_mat,
												 projectionMatrix=proj_mat,
												 renderer=p.ER_BULLET_HARDWARE_OPENGL)


		rgb = np.array(rgb,dtype=np.uint8)[:,:,:3][:,:,::-1].copy()
		cv2.imwrite(root+'/ones/%s_all.png'%step, rgb)


		state = 255 * state

		state_global = state.astype(np.uint8)[3:,:,:].transpose(1,2,0)[:,:,::-1].copy()

		state_local = state.astype(np.uint8)[:3,:,:].transpose(1,2,0)[:,:,::-1].copy()

		cv2.imwrite(root+'/global/%d_global.png'%step, state_global)   
		cv2.imwrite(root+'/local/%s_lcoal.png'%step, state_local)	

	# def get_action_from_obs(self,obs_target):
	# 	error = np.array(obs_target) - np.array(world_target_position)
	# 	dist = (error[0] ** 2 + error[1] ** 2) ** 0.5
	# 	if dist>=0.01:
	# 		if error[1]<-0.01:
	# 			rgb_state, reward, done = self.step([-20, -20, -20, -20, 0, 0, 0])
	# 		elif error[1]>0.01:
	# 			rgb_state, reward, done = self.step([20, 20, 20, 20, 0, 0, 0])
	# 		else:
	# 			if error[0] > 0:
	# 				rgb_state, reward, done = self.step([-40,-40,40,40,0,0,0])
	# 			else:
	# 				rgb_state, reward, done = self.step([40,40,-40,-40,0,0,0])
	# 		return rgb_state, reward, done
	# 	else:
	# 		if not self.done:
	# 			rgb_state, reward, done = self.start_gripper()
	# 		else:
	# 			rgb_state, reward, done = self.step([0, 0, 0, 0, 0, -5, 5])
	# 		return rgb_state, reward, done

	def get_action_from_obs(self,obs_target):
		if abs(obs_target[2]-obs_target[0])<0.7:
			obs_pos = [obs_target[0]/2+obs_target[2]/2,obs_target[1]/2+obs_target[3]/2]
			if not self.robot.arm_down:
				error_1 = np.array(obs_pos) - np.array(world_target_local)
			else:
				error_1 = np.array(obs_pos) - np.array(world_target_local_down)
		else:
			error_1 = np.zeros(2)
		if abs(obs_target[6]-obs_target[4])<0.7:
			obs_pos = [obs_target[4]/2+obs_target[6]/2,obs_target[5]/2+obs_target[7]/2]
			error_2 = np.array(obs_pos) - np.array(world_target_global)
		else:
			error_2 = np.zeros(2)
		if self.robot.arm_down:
			error = 0.8*error_1 + 0.2*error_2
		else:
			error = 0.2*error_1 + 0.8*error_2
		dist = (error[0] ** 2 + error[1] ** 2) ** 0.5
		print('\r Step:%3d Dist: %.4f   Error:%.4f  %.4f'%(self.step_num,dist,error[0],error[1]),end='')
		if (dist>=0.015 and dist <=0.12) or error[1]<0.0008:# 误差在0.04以内都可抓取，这里设置为0.015
			if not self.robot.arm_down:
				self.down_gripper()
			if error[0]>0.015:
				self.step([-10, -10, -10, -10, 0, 0, 0])
			elif error[0]<-0.015:
				self.step([10, 10, 10, 10, 0, 0, 0])
			else:
				if error[1] < 0:
					self.step([-20,-20,20,20,0,0,0])
				else:
					self.step([20,20,-20,-20,0,0,0])
		elif dist>0.12:# 误差较大时，抓起升起
			if dist > 0.25: # 不同视角误差不同，避免降下爪子误差增大，再抬上去的情况
				if self.robot.arm_down:
					self.up_gripper()
			if error[0]>0.03:
				self.step([-80, -80, -80, -80, 0, 0, 0])
			elif error[0]<-0.03:
				self.step([60, 60, 60, 60, 0, 0, 0])
			else:
				if error[1] < 0:
					self.step([-100,-100,100,100,0,0,0])
				else:
					self.step([100,100,-100,-100,0,0,0])
		else:
			if not self.done:
				self.start_gripper()
				self.robot.arm_down = False

	def down_gripper(self):
		for i in range(30):
			self.step([0,0,0,0,5,10,-10])
		self.robot.arm_down = True

	def up_gripper(self):
		for i in range(20):
			self.step([0,0,0,0,-1,10,-10])
		self.robot.arm_down = False

	def start_gripper(self):
		for i in range(5):
			self.step([0,0,0,0,2,5,-5])
		for i in range(20):
			self.step([0,0,0,0,0,-5,5])
		for i in range(30):
			self.step([0,0,0,0,-1,-5,5])
		target_position, _ = p.getBasePositionAndOrientation(self.object)
		if target_position[2]>=0.06:
			print('    Success  \n***************************')
			self.done = True



	# def auto_move(self):
	# 	position, orient = p.getBasePositionAndOrientation(self.robot.robot_id)
	# 	target_position, _ = p.getBasePositionAndOrientation(self.object)
	# 	matrix = np.array(p.getMatrixFromQuaternion(orient, physicsClientId=self.server_id)).reshape(3, 3)
	# 	orient = np.matmul(matrix, np.array([1,0,0]))
	# 	end_eff_position = np.array(position) + np.matmul(matrix, np.array(base_target_position))
	# 	error = np.array([target_position[i] - end_eff_position[i] for i in range(3)])
	# 	dist = (error[0] ** 2 + error[1] ** 2) ** 0.5
	# 	angle = self._GetClockAngle(orient[:2], error[:2])
	# 	ncount = 0
	# 	while dist>=0.01 and ncount < self.max_step:
	# 		ncount +=1
	# 		if angle <-0.005:
	# 			self.step([-10,-10,-10,-10,0,0,0])
	# 		elif angle > 0.005:
	# 			self.step([10,10,10,10,0,0,0])
	# 		elif abs(angle) < 0.005:
	# 			if error[0] > 0:
	# 				self.step([-20,-20,20,20,0,0,0])
	# 			else:
	# 				self.step([20,20,-20,-20,0,0,0])
	# 		print(self.step_num, dist, angle)
	# 		position, orient = p.getBasePositionAndOrientation(self.robot.robot_id)
	# 		target_position, _ = p.getBasePositionAndOrientation(self.object)
	# 		matrix = np.array(p.getMatrixFromQuaternion(orient, physicsClientId=self.server_id)).reshape(3, 3)
	# 		orient = np.matmul(matrix, np.array([1, 0, 0]))
	# 		end_eff_position = np.array(position) + np.matmul(matrix, base_target_position)
	# 		error = np.array([target_position[i] - end_eff_position[i] for i in range(3)])
	# 		dist = (error[0] ** 2 + error[1] ** 2) ** 0.5
	# 		angle = self._GetClockAngle(orient[:2], error[:2])
	# 	self.start_gripper()



	def reset(self):
		p.resetSimulation()
		self.__init__(self.server_id, self.urdf_path, self.urdf_object, self.max_step, self.mode, self.pure)
		state,_,_ = self.robot.step([0,0,0,0,0,20,-20])
		return state

	def close(self):
		p.disconnect(self.server_id)
