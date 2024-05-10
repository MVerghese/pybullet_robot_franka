from collections import deque
import numpy as np
np.seterr(all='raise')
import pybullet as pb
from pybullet_robot.worlds import add_PyB_models_to_path
from pybullet_robot.worlds.simple_world import SimpleWorld
from pybullet_robot.robots.bullet_panda.panda_robot import PandaArm
from pybullet_robot.controllers.os_hyb_ctrl import OSHybridController
import os 
import time
import matplotlib.pyplot as plt
import threading
import imageio
from tqdm import tqdm 
from PIL import Image

import torch
import torchvision.transforms as transforms
from PPO import PPO
from lavila_encoding import LaViLa_Interface, read_video_frames

# import warnings
# warnings.filterwarnings("error")

assets_path = "./assets/"
object_info = {}
object_info['frying_pan'] = {'path': 'assets/frying_pan.urdf', 'scale': 0.5, 'ref_obj': True,'grasp_obj': False,'init_pos':[.5,0.,.5]}
object_info['spatula'] = {'path': 'assets/spatula.urdf', 'scale': 0.5, 'ref_obj': False,'grasp_obj': True,'init_pos':[.5,0.,.6], 'gripper_offset': [.07,0.,-0.09],'obj_rot':[0,np.pi,np.pi/2]}
task_objs = {}
task_objs['Stir'] = ['frying_pan','spatula','peppers']

def create_peppers(count):
	pepper_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.005,0.005,0.005])
	red_pepper_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005,0.005,0.003], rgbaColor=[1,0,0,1])
	yellow_pepper_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005,0.005,0.003], rgbaColor=[1,1,0,1])
	red_xyz = [.5,0.03,.7]
	yellow_xyz = [.5,-0.03,.7]
	red_positions = [[red_xyz[0],red_xyz[1],red_xyz[2]+0.01*i] for i in range(count)]
	yellow_positions = [[yellow_xyz[0],yellow_xyz[1],yellow_xyz[2]+0.01*i] for i in range(count)]
	red_peppers = pb.createMultiBody(baseMass=.01, baseCollisionShapeIndex=pepper_collision, baseVisualShapeIndex=red_pepper_visual, batchPositions=red_positions)
	yellow_peppers = pb.createMultiBody(baseMass=.01, baseCollisionShapeIndex=pepper_collision, baseVisualShapeIndex=yellow_pepper_visual, batchPositions=yellow_positions)
	print("Created peppers")
	print(red_peppers,yellow_peppers)
	return(red_peppers, yellow_peppers)


def create_fixed_constraint(obj1,obj2, obj1_link = -1, obj2_link = -1):
	"""
	Creates a fixed constraint between two objects using the current relative pose between them.
	"""
	# print(pb.getNumJoints(obj1))
	# print(obj1,obj1_link)
	# print(pb.getLinkState(obj1, obj1_link))
	# print(pb.getNumJoints(obj2))
	# print(obj2,obj2_link)
	# print(pb.getLinkState(obj2, obj2_link))
	if obj1_link == -1:
		w_obj1_pos, w_obj1_ori = pb.getBasePositionAndOrientation(obj1)
	else:
		w_obj1_pos, w_obj1_ori = pb.getLinkState(obj1, obj1_link)[:2]
	if obj2_link == -1:
		w_obj2_pos, w_obj2_ori = pb.getBasePositionAndOrientation(obj2)
	else:
		w_obj2_pos, w_obj2_ori = pb.getLinkState(obj2, obj2_link)[:2]

	obj2_w_pos, obj2_w_ori = pb.invertTransform(w_obj2_pos, w_obj2_ori)

	obj2_obj1_pos, obj2_obj1_ori = pb.multiplyTransforms(obj2_w_pos, obj2_w_ori, w_obj1_pos, w_obj1_ori)

	obj1_obj2_pos, obj1_obj2_ori = pb.invertTransform(obj2_obj1_pos, obj2_obj1_ori)

	return pb.createConstraint(obj1, obj1_link, obj2, obj2_link, pb.JOINT_FIXED, [0,0,0], obj1_obj2_pos, [0,0,0], parentFrameOrientation=obj1_obj2_ori)

class CookingSim:
	def __init__(self,obj_list):

		self.robot = PandaArm()
		self.robot.set_ft_sensor_at(7)

		add_PyB_models_to_path()

		pb.setGravity(0, 0, -9.81)


		plane = pb.loadURDF('plane.urdf')
		table = pb.loadURDF('table/table.urdf',
						useFixedBase=True, globalScaling=0.5)
		pb.resetBasePositionAndOrientation(table, [0.4, 0., 0.0], [0, 0, -0.707, 0.707])

		self.objects = {"plane": plane, "table": table}
		for obj in obj_list:
			if obj in object_info.keys():
				info = object_info[obj]
				object_path = assets_path + obj +'/' + obj + '.urdf'
				print(object_path)
				self.objects[obj] = pb.loadURDF(object_path,info['init_pos'],
											useFixedBase=False, globalScaling=info['scale'])
			elif obj == "peppers":
				red_peppers, yellow_peppers = create_peppers(10)
				self.objects['red_peppers'] = red_peppers
				self.objects['yellow_peppers'] = yellow_peppers

		self.nouns = [None]*(len(self.objects)+1)
		self.nouns[0] = "robot"
		for obj in self.objects.keys():
			print(obj)
			if isinstance(self.objects[obj], int):
				self.nouns[self.objects[obj]] = obj
			elif isinstance(self.objects[obj], list):
				for i in range(len(self.objects[obj])):
					self.nouns[self.objects[obj][i]] = obj + str(i)


		



		self.world = SimpleWorld(self.robot, self.objects)
		pb.changeDynamics(self.world.objects.table, -1,
						  lateralFriction=0.1, restitution=0.9)

		self.cam_view_matrix = pb.computeViewMatrix([.5,0.,1.], [.5,0.,.5], [1.,0.,0.])
		self.cam_proj_matrix = pb.computeProjectionMatrixFOV(fov=60, aspect=1., nearVal=0.01, farVal=100.)

		self.controller = OSHybridController(self.robot)
		for noun in self.nouns:
			if noun in object_info.keys() and object_info[noun]['ref_obj']:
				create_fixed_constraint(self.objects[noun], self.objects['table'])
			elif noun in object_info.keys() and object_info[noun]['grasp_obj']:
				ee_pos, ee_ori  = pb.getLinkState(0,7)[:2]
				ee_pos = np.array(ee_pos)
				ee_pos += object_info[noun]['gripper_offset']
				num_bot_joints = pb.getNumJoints(0)
				pb.resetJointState(0,num_bot_joints-1,0.04)
				pb.resetJointState(0,num_bot_joints-2,0.04)
				pb.resetBasePositionAndOrientation(self.objects[noun], ee_pos, pb.getQuaternionFromEuler(object_info[noun]['obj_rot']))
				create_fixed_constraint(0, self.objects[noun], obj1_link = 7)
			


		self.controller.start_controller_thread()
		pb.setJointMotorControl2(0,num_bot_joints-1,pb.TORQUE_CONTROL,force=-20)
		pb.setJointMotorControl2(0,num_bot_joints-2,pb.TORQUE_CONTROL,force=-20)

		self.init_world_state = pb.saveState()

	def reset_world_state(self):
		pb.restoreState(self.init_world_state)

	def get_camera_info(self):
		return(pb.getCameraImage(640, 480, self.cam_view_matrix, self.cam_proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL))

	def determine_obj_contact(self, obj1, obj2):
		contact_points = pb.getContactPoints(obj1, obj2)
		return(contact_points)

	def get_vis_object_pose(self, camera_image, ref_obj, act_obj):
		ref_obj_id = self.nouns.index(ref_obj)
		if ref_obj_id  == -1:
			raise Exception("Reference object not found")
		ref_obj_mask = camera_image[4] == ref_obj_id
		act_obj_id = self.nouns.index(act_obj)
		if act_obj_id  == -1:
			raise Exception("Acting object not found")
		act_obj_mask = camera_image[4] == act_obj_id
		ref_obj_points = np.argwhere(ref_obj_mask)
		act_obj_points = np.argwhere(act_obj_mask)
		# check if it is an empty slice
		assert ref_obj_points.size > 0, "Reference object not found in image"
		ref_obj_pose = np.mean(ref_obj_points, axis=0)
		assert act_obj_points.size > 0, "Reference object not found in image"
		act_obj_pose = np.mean(act_obj_points, axis=0)

		#compute covariance of act_obj_points
		act_obj_cov = np.cov(act_obj_points.T)
		eigvals,eigvecs = np.linalg.eig(act_obj_cov)


		return(act_obj_pose - ref_obj_pose, eigvecs[:,0])


	def primitive_contact_force():
		pass

	def run_test_sim(self):
		goal_pos, goal_ori = self.robot.ee_pose()
		self.controller.start_controller_thread()
		z_traj = np.linspace(goal_pos[2], 0.3, 550)
		slow_rate = 100.
		try:
			i = 0
			while i < z_traj.size:
				now = time.time()

				ee_pos, _ = self.robot.ee_pose()
				wrench = self.robot.get_ee_wrench(local=False)
				# print wrench
				if abs(wrench[2]) >= 20.:
					print("Force threshold reached")
					break

				goal_pos[2] = z_traj[i]

				self.controller.update_goal(goal_pos, goal_ori)

				# fx_deque.append(wrench[0])
				# fy_deque.append(wrench[1])
				# fz_deque.append(wrench[2])

				# camera_image = pb.getCameraImage(640, 480, cam_view_matrix, cam_proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
				# print(np.shape(camera_image[4]))
				# plt.imshow(camera_image[4])
				# plt.show()
				camera_info = self.get_camera_info()
				print(self.get_vis_object_pose(camera_info, 'frying_pan', 'spatula'))
				
				elapsed = time.time() - now
				sleep_time = (1./slow_rate) - elapsed
				if sleep_time > 0.0:
					time.sleep(sleep_time)

				print("ITERATION: ",i)


				i += 1
			else:
				print("Force threshold not reached")
		except KeyboardInterrupt:
			print("exiting")

		finally:
			self.controller.stop_controller_thread()

class StirReward:
	def __init__(self, sim):
		self.stir_points = []
		self.sim = sim
		self.first_call = True
		self.init_dist = 1.

	def check_pepper_pan_contact(self):
		red_pepper_contact = np.zeros(len(self.sim.objects['red_peppers']),dtype=int)
		for i in range(len(self.sim.objects['red_peppers'])):
			contact_points = self.sim.determine_obj_contact(self.sim.objects['red_peppers'][i], self.sim.objects['frying_pan'])
			if len(contact_points) > 0:
				red_pepper_contact[i] = 1

		yellow_pepper_contact = np.zeros(len(self.sim.objects['yellow_peppers']),dtype=int)
		for i in range(len(self.sim.objects['yellow_peppers'])):
			contact_points = self.sim.determine_obj_contact(self.sim.objects['yellow_peppers'][i], self.sim.objects['frying_pan'])
			if len(contact_points) > 0:
				yellow_pepper_contact[i] = 1

		return(red_pepper_contact, yellow_pepper_contact)

	def compute_reward(self):
		red_pepper_contact, yellow_pepper_contact = self.check_pepper_pan_contact()
		red_pepper_pos = [pb.getBasePositionAndOrientation(pepper)[0] for pepper in self.sim.objects['red_peppers']]
		yellow_pepper_pos = [pb.getBasePositionAndOrientation(pepper)[0] for pepper in self.sim.objects['yellow_peppers']]
		# compute the average distance between red and yellow peppers
		red_pepper_pos = np.array(red_pepper_pos)
		yellow_pepper_pos = np.array(yellow_pepper_pos)
		# Only keep the peppers that are in contact with the pan
		red_pepper_pos = red_pepper_pos[red_pepper_contact == 1]
		yellow_pepper_pos = yellow_pepper_pos[yellow_pepper_contact == 1]
		
		bad_action_done = False
		try:
			red_pepper_pos = np.mean(red_pepper_pos, axis = 0)
		except:
			return ( self.init_dist-self.init_dist, True )
			# assert False, f"{red_pepper_pos.shape}\n{red_pepper_pos}"
		try:
			yellow_pepper_pos = np.mean(yellow_pepper_pos, axis = 0)
		except:
			return ( self.init_dist-self.init_dist, True )
			# assert False, f"{yellow_pepper_pos.shape}\n{yellow_pepper_pos}"
		dist = np.linalg.norm(red_pepper_pos - yellow_pepper_pos)
		if self.first_call:
			self.init_dist = dist
			self.first_call = False
		reward = self.init_dist-dist
		return (reward, bad_action_done )

reward_models = {}
reward_models['Stir'] = StirReward

class CookingEnv:
	def __init__(self, task_id = "Stir"):
		self.task_id = task_id
		self.task = None
		self.task_objs = task_objs[task_id]
		self.sim = CookingSim(self.task_objs)
		self.reward = reward_models[task_id](self.sim)
		self.timeout = 2000

	def gen_obs(self,return_cam = True, return_proprioception = True):
		obs = {}
		if return_cam:
			obs['camera'] = self.sim.get_camera_info()
		if return_proprioception:
			obs['proprioception'] = self.sim.robot.ee_pose()
		return(obs)

	def step(self,action):
		# Action is a 3 tuple of (goal_pos, goal_ori, force_axes)
		goal_pos, goal_ori, force_axes = action
		self.sim.controller.update_goal(goal_pos, goal_ori, goal_force = force_axes[:3], goal_torque = force_axes[3:])
		# add force axes
		unit_force_axes = [1 if axis != 0 else 0 for axis in force_axes]
		# print(unit_force_axes)
		self.sim.controller.change_ft_directions(unit_force_axes)
		obs = self.gen_obs()
		reward, bad_action_done = self.reward.compute_reward()
		if not bad_action_done:
			done = self.timeout == 0
		else:
			# terminate early if rewards is nan
			done = True
		
		self.timeout -= 1
		return(obs, reward, done, bad_action_done, {})

	def reset(self):
		obs = self.gen_obs()
		self.sim.reset_world_state()
		return(obs)

	def close(self):
		self.sim.controller.stop_controller_thread()

def create_task(task_id):
	pass

def image2state(pil_image, transform):
	# use transform 
	return transform(pil_image)
	

def run_experiment():
	# Initialization
	device = 'cuda:3'
	env = CookingEnv()
	IDX = 5
	exp_directory = f'./../output/{IDX}'
	loss_path = os.path.join(exp_directory, 'loss.txt')
	return_path = os.path.join(exp_directory, 'return.txt')
	if not os.path.exists(exp_directory):
		os.makedirs(exp_directory)
		
	# hyperparam
	episode_len = 300 # 1000
	ep_timeout = 3 * 60 # 5min
	max_training_timesteps =  int(1.5e5)
	done = False
 
	# initialize a PPO agent
	action_scalar = 0.1				# reduce action delta by a sclar
	action_dim = 3
	update_timestep = episode_len * 3       # update policy every epoch
	K_epochs = 8               # update policy for K epochs in one PPO update
	has_continuous_action_space=True
	eps_clip = 0.2          # clip parameter for PPO
	gamma = 0.99            # discount factor
	action_std = 0.6		# use for exploration
	action_std_decay_rate = 0.05		# linearly decay action_std (action_std = action_std - action_std_decay_rate)
	min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
	action_std_decay_freq = int(1.5e4)  # action_std decay frequency (in num timesteps)
 
	lr_actor = 0.0003       # learning rate for actor network
	lr_critic = 0.001       # learning rate for critic network

	random_seed = 0         # set random seed if required (0 = no random seed)
	torch.manual_seed(random_seed)
	ppo_agent = PPO(action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std, device)
	transform_img = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
 
	# initilaize reward models
	use_lavila = True # True: reward = lavila reward + reward
	lavila_only = True # True reward = lavila reward
	# RefName = 'STIR_P01_01_39343_39657'
	RefName = 'STIR_P28_103_47308_47458'
	if use_lavila:
		frame_num = 16
		ckpt_path = '/home/yiqiw2/experiment/RoboNLP/lavila_ckpt/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth'
		lavila_inter = LaViLa_Interface(ckpt_path, cuda_device= device, clip_length=frame_num )
		# read .npy from npy_path
		npy_path = f'/home/yiqiw2/experiment/RoboNLP/VideoReferences/{RefName}.npy'
		with open(npy_path, 'rb') as f:
			ref_embedding = torch.from_numpy( np.load(f) ).float().unsqueeze(0).numpy()
   
	# logging 
	best_ep_return = 0
	episode = 0
	time_step = 0
	while time_step <= max_training_timesteps:
		episode += 1
		# logging
		frames = []
		rewards = [] # only collect environmental rewards (no lavila rewards)
		obs = env.reset()
		
		state = Image.fromarray(obs['camera'][2]).convert('RGB').resize((320,240))

		state = image2state(state, transform_img)
		pbar = tqdm(range(episode_len) )
		start_time = time.time()
		for step in pbar:
			if time.time() - start_time > ep_timeout:
				break
			
			predicted_action = ppo_agent.select_action(state)
			predicted_action *= action_scalar
			new_action_3d = obs['proprioception'][0] + predicted_action
			# assert False, f"{predicted_action }\n{init_action[0]}"
			action = [ new_action_3d, obs['proprioception'][1], np.zeros(6,dtype=int)]

			try:
				obs, reward, done, bad_done, _ = env.step(action)
			except:
				# print( f"Bad action: {action}" )
				assert False, f"bad action: {action}"

			# pil image as state 
			state = Image.fromarray(obs['camera'][2]).convert('RGB').resize((320,240))
			rgb = np.array( state  )
			frames.append(np.uint8(rgb))
   			# image dim: (3, 240, 320)
			state = image2state(state, transform_img)
			# lavila rewards is excluded from the reward logging
			rewards.append( reward )
			# print('step:', step)
			if use_lavila and ( done or bad_done or step == episode_len-1):
				
				sampled_frames = read_video_frames(frames)
				sampled_frames = torch.from_numpy(sampled_frames).float().to(device)
				rollout_embedding = lavila_inter.video_feature( sampled_frames).detach().cpu().numpy()
				vid_reward = ( rollout_embedding @ ref_embedding.T  ).flatten()[0]
				if not  lavila_only:
					reward += vid_reward
				else:
					reward = vid_reward
				
			time_step +=1

			# lavila rewards is included for training
			if not bad_done:
				# saving reward and is_terminals
				ppo_agent.buffer.rewards.append(reward)
				ppo_agent.buffer.is_terminals.append(done)

			# update PPO agent
			loss = None
			if time_step % update_timestep == 0:
				loss = ppo_agent.update()
				with open(loss_path, 'a') as f:
					f.write(f'{episode}:{loss}\n')
		
			# update progress bar, encouter invalid value
			pbar.set_postfix(
	   			{'loss':loss, 'sum_reward': round(np.sum(rewards),3), 'episode': episode})
			# if continuous action space; then decay action std of ouput action distribution
			if has_continuous_action_space and time_step % action_std_decay_freq == 0:
				ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

			
			if done or bad_done:
				break
		try:
			ep_return = np.sum(rewards )
		except ValueError as e:
			assert False, f"{e}"

		with open(return_path, 'a') as f:
			f.write(f'{episode}:{ep_return}\n' )
		if ep_return >best_ep_return :
			# log episode view
			best_ep_return = ep_return
			output_gif_file = f'best_output_episode_{episode}_{round(ep_return,3)}.gif'
			# remove redundent and add new
			gif_names = [ name for name in os.listdir(exp_directory) if '.gif' in name ]
			if len(gif_names) > 2:
				for name in gif_names:
					if f'best_output_episode' in name:
						os.remove(os.path.join(exp_directory, name))
			output_path = os.path.join(exp_directory, output_gif_file)
			imageio.mimsave(output_path, frames, duration=0.1, loop = 0)
   
			# log model
			output_weight_file = 'best_model.pth'
			checkpoint_path = os.path.join(exp_directory, output_weight_file)
			ppo_agent.save(checkpoint_path)

		output_gif_file = f'latest_output_episode_{episode}.gif'
		# remove redundent and add new
		gif_names = [ name for name in os.listdir(exp_directory) if '.gif' in name and 'latest' in name ]
		# print('gif_names:', gif_names )
		if len(gif_names) > 1:
			for name in gif_names:
				# print('name:', name, f'latest_output_episode_{episode}' in name)
				if f'latest' in name:
					os.remove(os.path.join(exp_directory, name))
		output_path = os.path.join(exp_directory, output_gif_file)
		imageio.mimsave(output_path, frames, duration=0.1, loop = 0)

	env.close()
 
	output_file = './output/final_output.gif'
	imageio.mimsave(output_file, frames, duration=0.1)

if __name__ == '__main__':
	run_experiment()