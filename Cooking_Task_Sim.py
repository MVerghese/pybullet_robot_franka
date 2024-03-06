from collections import deque
import numpy as np
import pybullet as pb
from pybullet_robot.worlds import add_PyB_models_to_path
from pybullet_robot.worlds.simple_world import SimpleWorld
from pybullet_robot.robots.bullet_panda.panda_robot import PandaArm
from pybullet_robot.controllers.os_hyb_ctrl import OSHybridController

import time
import matplotlib.pyplot as plt
import threading

assets_path = "/home/mverghese/MBLearning/pybullet_robot/assets/"
object_info = {}
object_info['frying_pan'] = {'path': 'assets/frying_pan.urdf', 'scale': 0.5, 'ref_obj': True,'grasp_obj': False,'init_pos':[.5,0.,1.]}
object_info['spatula'] = {'path': 'assets/spatula.urdf', 'scale': 0.5, 'ref_obj': False,'grasp_obj': True,'init_pos':[.5,0.,1.1], 'gripper_offset': [.07,0.,-0.09],'obj_rot':[0,np.pi,np.pi/2]}

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

class CookingTask:
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
			info = object_info[obj]
			object_path = assets_path + obj +'/' + obj + '.urdf'
			print(object_path)
			self.objects[obj] = pb.loadURDF(object_path,info['init_pos'],
										useFixedBase=False, globalScaling=info['scale'])

		self.nouns = [None]*(len(self.objects)+1)
		self.nouns[0] = "robot"
		for obj in self.objects.keys():
			self.nouns[self.objects[obj]] = obj

		



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

	def get_camera_info(self):
		return(pb.getCameraImage(640, 480, self.cam_view_matrix, self.cam_proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL))

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
		ref_obj_pose = np.mean(ref_obj_points, axis=0)
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


def create_task(task_id):
	pass



def run_experiment():
	task = CookingTask(['frying_pan','spatula'])
	task.run_test_sim()

if __name__ == '__main__':
	run_experiment()