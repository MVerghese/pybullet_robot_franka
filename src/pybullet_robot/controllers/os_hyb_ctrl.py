import numpy as np
from .utils import quatdiff_in_euler
from .os_controller import OSControllerBase
from .ctrl_config import OSHybConfig


class OSHybridController(OSControllerBase):

    def __init__(self, robot, config=OSHybConfig, gravity_comp = True, **kwargs):

        OSControllerBase.__init__(self, robot=robot, config=config, **kwargs)

        # print("PARAMS")
        # print(config['P_f'])
        # print(config['P_tor'])
        # print(config['I_f'])
        # print(config['I_tor'])
        # print(config['null_stiffness'])
        # print(config['windup_guard'])
        # print(config['ft_directions'])
        # print("END PARAMS")

        self._P_ft = np.diag(np.append(config['P_f'],config['P_tor']))
        self._I_ft = np.diag(np.append(config['I_f'],config['I_tor']))

        self._null_Kp = np.diag(config['null_stiffness'])

        self._windup_guard = np.asarray(config['windup_guard']).reshape([6,1])

        self.change_ft_directions(np.asarray(config['ft_directions'], int))

        self._gravity_comp = gravity_comp
        self._velocity_damping = .5

    def change_ft_directions(self, dims):
        self._mutex.acquire()
        self._ft_dir = np.diag(dims)
        self._pos_dir = np.diag([1, 1, 1, 1, 1, 1]) ^ self._ft_dir
        self._I_term = np.zeros([6, 1])
        self._mutex.release()

    def update_goal(self, goal_pos, goal_ori, goal_force = np.zeros(3), goal_torque = np.zeros(3)):
        self._mutex.acquire()
        self._goal_pos = np.asarray(goal_pos).reshape([3, 1])
        self._goal_ori = np.asarray(goal_ori)
        self._goal_ft = -np.append(np.asarray(goal_force), np.asarray(goal_torque)).reshape([6, 1])
        self._mutex.release()

    def _compute_cmd(self):
        """
        Actual control loop. Uses goal pose from the feedback thread
        and current robot states from the subscribed messages to compute
        task-space force, and then the corresponding joint torques.
        """
        ## MOTION CONTROL
        curr_pos, curr_ori = self._robot.ee_pose()

        delta_pos = self._goal_pos - curr_pos.reshape([3, 1])
        delta_ori = quatdiff_in_euler(
            curr_ori, self._goal_ori).reshape([3, 1])

        curr_vel, curr_omg = self._robot.ee_velocity()

        # Desired task-space motion control PD law
        F_motion = self._pos_dir.dot(np.vstack([self._P_pos.dot(delta_pos), self._P_ori.dot(delta_ori)]) - \
                                     np.vstack([self._D_pos.dot(curr_vel.reshape([3, 1])),
                                                self._D_ori.dot(curr_omg.reshape([3, 1]))]))
        # print("Position control values: ", F_motion.flatten())

        ## FORCE CONTROL
        last_time = self._last_time if self._last_time is not None else self._sim_time
        current_time = self._sim_time
        delta_time = max(0.,current_time - last_time)

        curr_ft = self._robot.get_ee_wrench(local=False).reshape([6, 1])
        # print("FT: ", curr_ft.flatten())
        # curr_ft[5,0] = 0.0 # remove torque about z-axis

        delta_ft = self._ft_dir.dot(self._goal_ft - curr_ft)
        self._I_term += delta_ft * delta_time
        # print np.diag(self._pos_dir), np.diag(self._ft_dir)
        self._I_term[self._I_term+self._windup_guard < 0.] = -self._windup_guard[self._I_term+self._windup_guard < 0.]
        self._I_term[self._I_term-self._windup_guard > 0.] = self._windup_guard[self._I_term-self._windup_guard > 0.]

        # Desired task-space force control PI law
        F_force = self._P_ft.dot(delta_ft) + self._I_ft.dot(self._I_term) + self._goal_ft

        # print("Force control values: ", F_force.flatten())
        
        F = F_motion - F_force # force control is subtracted because the computation is for the counter force

        error = np.asarray([(np.linalg.norm(self._pos_dir[:3, :3].dot(delta_pos))), np.linalg.norm(self._pos_dir[3:, 3:].dot(delta_ori)),
                        np.linalg.norm(delta_ft[3:]), np.linalg.norm(delta_ft[3:])])

        J = self._robot.jacobian()
        # print("Jacobian shape: ", J.shape)
        J = J[:,:7]

        self._last_time = current_time

        cmd = np.dot(J.T, F)

        # print(self._null_Kp.shape)

        null_space_filter = self._null_Kp.dot(
            np.eye(7) - J.T.dot(np.linalg.pinv(J.T, rcond=1e-3)))

        # print(self._robot._tuck)
        # print(self._robot.angles())
        cmd += null_space_filter.dot((self._robot._tuck-self._robot.angles()[:7]).reshape([7,1]))
        # print null_space_filter.dot(
            # (self._robot._tuck-self._robot.angles()).reshape([7, 1]))
        # cmd = np.zeros((7,1))

        if self._gravity_comp:
            joint_states = self._robot.get_joint_state()
            # joint_states = [np.zeros(9),np.zeros(9)]
            grav_comp_term = self._robot.inverse_dynamics(joint_states[0], joint_states[1]*self._velocity_damping, np.zeros(9))[:7]
            # print("Grav comp: ", grav_comp_term)
            cmd += grav_comp_term.reshape([7,1])

        # joint torques to be commanded

        # print("COMMAND TORQUES: ", cmd.flatten())
        return cmd, error

    def _initialise_goal(self):
        self._last_time = None
        self._I_term = np.zeros([6,1])
        self.update_goal(self._robot.ee_pose()[0], self._robot.ee_pose()[1], np.zeros(3), np.zeros(3))
