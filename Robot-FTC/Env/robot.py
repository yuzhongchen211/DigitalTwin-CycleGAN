import pybullet as p
import pybullet_data
import numpy as np
import random
import cv2
import time


class Camera(object):
	def __init__(self, robot_id, server_id, fov=49.1, aspect=1.0, nearVal=0.03, farVal=100):
		self.robot_id = robot_id
		self.server_id = server_id
		self.fov = fov
		self.aspect = aspect
		self.nearVal = nearVal
		self.farVal = farVal

	def capture_image(self, width=320, height=180):
		rgb_local,bbox_1 = self.get_camera_local(width, height)
		rgb_global,bbox_2 = self.get_camera_global(width, height)
		rgb_1 = np.array(rgb_local).transpose([2,0,1])[:3] / 255.0
		rgb_2 = np.array(rgb_global).transpose([2,0,1])[:3] / 255.0

		return np.vstack((rgb_1, rgb_2)),bbox_1,bbox_2

	def get_camera_global(self, width=320, height=180):
		result = [1.0, 1.0, .0, .0]
		camera_info_1 = p.getLinkState(self.robot_id, 9)
		camera_pos = np.array(camera_info_1[0])
		matrix = np.array(p.getMatrixFromQuaternion(camera_info_1[1], physicsClientId=self.server_id)).reshape(3, 3)
		camera_target_pos = camera_pos + np.matmul(matrix, np.array([0.88, 0, -0.36]))
		camera_up_vector = np.matmul(matrix, np.array([0.36, 0, 0.88]).reshape(3, 1))
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
		time_point = str(time.time())
		seg = np.array(seg,dtype=np.uint8)
		ret, thresh = cv2.threshold(seg, 1, 255, 0) 
		bboxs = self.mask_find_bboxs(thresh)
		for b in bboxs:
			if b[4]>1600 or abs(b[2]-b[3])>30 or (b[0]+b[1])<10: continue
			# x0, y0 = b[0], b[1]
			# x1 = b[0] + b[2]
			# y1 = b[1] + b[3]
			# result = [x0/width,y0/height,x1/width,y1/height]
			x0, y0 = b[0], b[1]
			x1 = b[0] + b[2]
			y1 = b[1] + b[3]
			# print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
			temp_result = [x0/width,y0/height,x1/width,y1/height]
			# print('*'*8)
			# print(temp_result,result)
			if (temp_result[0]+temp_result[1])<(result[0]+result[1]):
				result[0],result[1] = temp_result[0],temp_result[1]
			if (temp_result[2]+temp_result[3])>(result[2]+result[3]):
				result[2],result[3] = temp_result[2],temp_result[3]
			# print(result)
			# print('#'*8)
		# start_point, end_point = (int(result[0]*320),int(result[1]*180)),(int(result[2]*320),int(result[3]*180))
		# color = (0, 0, 255) # Red color in BGR；红色：rgb(255,0,0)
		# thickness = 1 # Line thickness of 1 px 
		# #mask_BGR = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) # 转换为3通道图，使得color能够显示红色。
		# mask_BGR = np.array(rgb,dtype=np.uint8)
		# mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
		# if random.random()<0.1: 
		# 	cv2.imwrite('temp/%s_bboxs_global.png'%time_point, mask_bboxs)
		return rgb,result
	def mask_find_bboxs(self,mask):
		retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
		stats = stats[stats[:,4].argsort()]
		return stats

	def get_camera_local(self, width=320, height=180):
		result = [1.0, 1.0, .0, .0]
		camera_info_1 = p.getLinkState(self.robot_id, 7)
		camera_pos = np.array(camera_info_1[0])
		matrix = np.array(p.getMatrixFromQuaternion(camera_info_1[1], physicsClientId=self.server_id)).reshape(3, 3)
		camera_target_pos = camera_pos + np.matmul(matrix, np.array([1.625, 0.097, -0.31]))
		camera_up_vector = np.matmul(matrix, np.array([1.625, 0.097, 0]))
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
		time_point = str(time.time())
		seg = np.array(seg,dtype=np.uint8)
		ret, thresh = cv2.threshold(seg, 1, 255, 0) 
		bboxs = self.mask_find_bboxs(thresh)
		for b in bboxs:
			if b[4]>1600 or abs(b[2]-b[3])>30 or (b[0]+b[1])<10: continue
			x0, y0 = b[0], b[1]
			x1 = b[0] + b[2]
			y1 = b[1] + b[3]
			temp_result = [x0/width,y0/height,x1/width,y1/height]
			if (temp_result[0]+temp_result[1])<(result[0]+result[1]):
				result[0],result[1] = temp_result[0],temp_result[1]
			if (temp_result[2]+temp_result[3])>(result[2]+result[3]):
				result[2],result[3] = temp_result[2],temp_result[3]
			# print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
		# start_point, end_point = (int(result[0]*320),int(result[1]*180)),(int(result[2]*320),int(result[3]*180))
		# color = (0, 0, 255) # Red color in BGR；红色：rgb(255,0,0)
		# thickness = 1 # Line thickness of 1 px 
		# #mask_BGR = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) # 转换为3通道图，使得color能够显示红色。
		# mask_BGR = np.array(rgb,dtype=np.uint8)
		# mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
		
		# """
		# # Displaying the image  
		# cv2.imshow('show_image', mask_bboxs) 
		# cv2.waitKey(0)
		# """
		# if random.random()<0.5:
		# 	cv2.imwrite('temp/%s_bboxs_lcoal.png'%time_point, mask_bboxs)
		return rgb,result

class Robot(Camera):
	def __init__(self, server_id, robot_urdf='./ftc_robot/urdf/ftc_robot.urdf'):
		self.robot_id = p.loadURDF(robot_urdf, basePosition=[0, 0, 0.262])
		self.server_id = server_id
		self.robot_urdf = robot_urdf
		super(Robot,self).__init__(self.robot_id,self.server_id)
		self.arm_down = False
		self.reset_arm()
	def reset_arm(self,down=False):
		if down:
			p.resetJointState(self.robot_id,5,0.25)
			p.resetJointState(self.robot_id,6,-0.25)
			p.resetJointState(self.robot_id,4,0.34)
			self.step([0,0,0,0,0,10,-10])
			self.arm_down=True
			
		else:
			p.resetJointState(self.robot_id,5,0.25)
			p.resetJointState(self.robot_id,6,-0.25)
			p.resetJointState(self.robot_id,4,0.2)
			self.step([0,0,0,0,0,10,-10])
			self.arm_down=False
			
	def step(self,action):
		if len(action) == 7:
			p.setJointMotorControlArray(
				bodyUniqueId=self.robot_id,
				jointIndices=[0, 1, 2, 3, 4, 5, 6],
				controlMode=p.VELOCITY_CONTROL,
				targetVelocities=action,
				forces=[200, 200, 200, 200, 10, 200, 200])
			p.stepSimulation()
		rgb,bbox_local,bbox_global = self.capture_image()
		return rgb,bbox_local,bbox_global


