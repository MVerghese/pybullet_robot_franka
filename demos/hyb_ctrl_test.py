from collections import deque
import numpy as np
import pybullet as pb
import pybullet_data
from pybullet_robot.worlds import add_PyB_models_to_path
from pybullet_robot.worlds.simple_world import SimpleWorld
from pybullet_robot.robots.bullet_panda.panda_robot import PandaArm
from pybullet_robot.controllers.os_hyb_ctrl import OSHybridController

import time
import matplotlib.pyplot as plt
import threading
import sys

def plot_thread():

    plt.ion()
    while True:
        plt.clf()
        plt.plot(fx_deque, 'r', label='x')
        plt.plot(fy_deque, 'g', label='y')
        plt.plot(fz_deque, 'b', label='z')
        plt.legend()
        plt.draw()
        plt.pause(0.000001)
        if done:
            break


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
    


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    print("Setting up panda arm")
    robot = PandaArm()
    robot.set_ft_sensor_at(7)

    add_PyB_models_to_path()

    pb.setGravity(0, 0, -9.81)

    plane = pb.loadURDF('plane.urdf')
    table = pb.loadURDF('table/table.urdf',
                        useFixedBase=True, globalScaling=0.5)
    print('Plane ID: ', plane)
    print('Table ID: ', table)

    pan = pb.loadURDF('../assets/frying_pan/frying_pan.urdf',[.5,0.,1.], useFixedBase = False, useMaximalCoordinates = True, flags=pb.URDF_USE_INERTIA_FROM_FILE, globalScaling=0.5)
    spatula = pb.loadURDF('../assets/spatula/spatula.urdf',[.5,0.,1.1], useFixedBase = False, useMaximalCoordinates = True, flags=pb.URDF_USE_INERTIA_FROM_FILE, globalScaling=.7)
    cube = pb.loadURDF('cube_small.urdf', useFixedBase=True, globalScaling=1.)
    pb.resetBasePositionAndOrientation(
        table, [0.4, 0., 0.0], [0, 0, -0.707, 0.707])

    cam_view_matrix = pb.computeViewMatrix([.5,0.,1.], [.5,0.,.5], [1.,0.,0.])
    cam_proj_matrix = pb.computeProjectionMatrixFOV(fov=60, aspect=1., nearVal=0.01, farVal=100.)




    objects = {'plane': plane,
               'table': table,
               'pan':pan,
               'spatula':spatula}

    world = SimpleWorld(robot, objects)
    pb.changeDynamics(world.objects.table, -1,
                      lateralFriction=0.1, restitution=0.9)
    slow_rate = 100.

    goal_pos, goal_ori = world.robot.ee_pose()
    print(goal_pos)

    controller = OSHybridController(robot)

    print("started")

    z_traj = np.linspace(goal_pos[2], 0.3, 550)

    # plot_t = threading.Thread(target=plot_thread)
    # fx_deque = deque([0],maxlen=1000)
    # fy_deque = deque([0],maxlen=1000)
    # fz_deque = deque([0],maxlen=1000)

    
    ee_pos, ee_ori  = pb.getLinkState(0,7)[:2]
    ee_pos = np.array(ee_pos)
    ee_pos += [.07,0.,-0.09]
    # spatula_pos, spatula_ori = pb.multiplyTransforms(ee_pos,ee_ori,[0,0,0],pb.getQuaternionFromEuler([np.pi/2,0,np.pi/2]))
    # spatula_pos = np.array(spatula_pos)
    # spatula_pos[2] += .5
    num_bot_joints = pb.getNumJoints(0)
    pb.resetJointState(0,num_bot_joints-1,0.04)
    pb.resetJointState(0,num_bot_joints-2,0.04)


    pb.resetBasePositionAndOrientation(spatula, ee_pos, pb.getQuaternionFromEuler([0,np.pi,np.pi/2]))
    print("spatula pos: ",pb.getBasePositionAndOrientation(spatula))
    create_fixed_constraint(0, world.objects.spatula, obj1_link = 7)
    create_fixed_constraint(world.objects.table, world.objects.pan)

    time.sleep(1)
    print("spatula pos: ",pb.getBasePositionAndOrientation(spatula))
    controller.start_controller_thread()
    pb.setJointMotorControl2(0,num_bot_joints-1,pb.TORQUE_CONTROL,force=-20)
    pb.setJointMotorControl2(0,num_bot_joints-2,pb.TORQUE_CONTROL,force=-20)



    

    done = False
    # plot_t.start()
    try:
        i = 0
        f_ctrl = True
        while i < z_traj.size:
            now = time.time()

            ee_pos, _ = world.robot.ee_pose()
            wrench = world.robot.get_ee_wrench(local=False)
            # print wrench
            if abs(wrench[2]) >= 20.:
                print("Force threshold reached")
                break

            goal_pos[2] = z_traj[i]

            controller.update_goal(goal_pos, goal_ori)

            # fx_deque.append(wrench[0])
            # fy_deque.append(wrench[1])
            # fz_deque.append(wrench[2])

            camera_image = pb.getCameraImage(640, 480, cam_view_matrix, cam_proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
            # print(np.shape(camera_image[4]))
            # plt.imshow(camera_image[4])
            # plt.show()
            
            elapsed = time.time() - now
            sleep_time = (1./slow_rate) - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

            print("ITERATION: ",i)


            i += 1
        else:
            print("Never reached force threshold for switching controller")
            f_ctrl = False
        
        if f_ctrl:
        
            print("Switching to force control along Z axis")
            y_traj = np.linspace(goal_pos[1], goal_pos[1]-0.2, 400)
            
            controller.change_ft_directions([0,0,1,0,0,0])
            target_force = -21.

            p_slider = pb.addUserDebugParameter('p_f',0.1,2.,controller._P_ft[2, 2])
            i_slider = pb.addUserDebugParameter('i_f',0.0,100.,controller._I_ft[2, 2])
            w_slider = pb.addUserDebugParameter('windup',0.0,100.,controller._windup_guard[2, 0])

            

            
            i = 0
            while i < y_traj.size:
                now = time.time()

                ee_pos, _ = world.robot.ee_pose()
                wrench = world.robot.get_ee_wrench(local=False)
                # print wrench
                goal_pos[1] = y_traj[i]

                controller._P_ft[2, 2] = pb.readUserDebugParameter(p_slider)
                controller._I_ft[2, 2] = pb.readUserDebugParameter(i_slider)
                controller._windup_guard[2, 0] = pb.readUserDebugParameter(w_slider)

                controller.update_goal(
                    goal_pos, goal_ori, np.asarray([0., 0., target_force]))

                # fx_deque.append(wrench[0])
                # fy_deque.append(wrench[1])
                # fz_deque.append(wrench[2])

                elapsed = time.time() - now
                sleep_time = (1./slow_rate) - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

                if i < y_traj.size-1:
                    i += 1

                camera_image = pb.getCameraImage(640, 480, cam_view_matrix, cam_proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
                

                # plt.clf()
                # camera_image = pb.getCameraImage(640, 480, cam_view_matrix, cam_proj_matrix, renderer=pb.ER_BULLET_HARDWARE_OPENGL)
                # plt.imshow(camera_image[2])
                # plt.draw()
                # plt.pause(0.000001)
    finally:


        controller.stop_controller_thread()
        done = True
        # plot_t.join()
