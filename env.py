import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import Formation_control_A as base
import random

class FormationEnv(gym.Env):
    def __init__(self, screen):
        self.WIDTH = 1980
        self.HEIGHT = 1080
        self.screen = screen
        self.dt = 0.005

        self.num_of_bots = 3
        self.robots = []
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2* self.num_of_bots,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(48,), dtype=np.float32)

        self.FORMATION_VELOCITY = np.array([0.5, 0.5])
        # try to make a triangle
        self.FORMATION_OFFSET = np.array([[0, 0], [20, 100], [100,50]])

        # adjacency matrix for a directed graph
        # self.A = np.array([
        #     [0, 1, 0],
        #     [0, 0, 1],
        #     [1, 0, 0]
        # ])

        self.A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
            ])

        self.Degree_matrix = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])

        #identity matrix
        self.I = np.eye(3)

        # Laplacian matrix 3x3
        self.L = self.I - self.Degree_matrix @ self.A
        print("Laplacian Matrix L:\n", self.L)

        self.I_m = np.eye(3)
  
        # Sytem dynamics matrices for a double integrator model
        # A0 is the state transition(from current state to next state x->dx) matrix for a single robot
        # B0 is the input matrix for a single robot
        # Each robot has a state of [x, dx, y, dy] (position and velocity in 2D)
        self.A0 = np.array([
            [0, 1, 0, 0],  # dx = vx
            [0, 0, 0, 0],  # dvx = ux (handled by B0)
            [0, 0, 0, 1],  # dy = vy
            [0, 0, 0, 0]   # dvy = uy (handled by B0)
        ])

        self.B0 = np.array([
            [0, 0],  # input doesn't affect position directly
            [1, 0],  # u_x affects dvx/dt
            [0, 0],  # input doesn't affect position directly
            [0, 1]   # u_y affects dvy/dt
        ])

        self.formation_anchor = 0
        #simple double integrator where only consider postion and velocity
        #we want to make sure robot's dynmics are indenpendent from each other,
        # since each robot has 4 states, we need to create a block diagonal matrix (3x3) x(4x4) for multiple robots
        #Av is the whole picture of the system dynamics
        #Bv is the input matrix for the whole system
        self.A_v = np.kron(np.eye(self.num_of_bots), self.A0) # This will create a 12x12 matrix 

        # For multiple robots (3 robots in this case), use the correct identity matrix for the number of states
        self.B_v = np.kron(np.eye(self.num_of_bots), self.B0)  # This will create a 12x6 matrix

        self.I_2m = np.eye(2 * 2)

        # Turning the Laplacian matrix into a 12x12 matrix to allow further calculations
        self.L1 = np.kron(self.L, np.eye(4))

        # we only have 2 g since our sytem is 2D
        # Gamma_1 = [
        #     [-g₁, -g₂,  0,    0   ],
        #     [ 0,    0,  -g₁, -g₂ ]
        # ]
        self.Gamma_1 = [
            [-50.0, -30.0,  0.0,  0.0], #gain control feedback for x-driection
            [  0.0,  0.0, -50.0, -30.0] #gain control feedback for y-driection
        ]
        # shape (2×4), controls x and y

        self.Gamma = np.kron(np.eye(self.num_of_bots), self.Gamma_1)  # shape (6×12)

        self.current_step = 0
        self.target = None
        self.desired_states = None
        self.last_action = 0
        self.prev_error = 0.0
        self.prev_target_dist = None
        
        # for improvement bonus
        self.prev_formation_error = None
        self.prev_tracking_error = None

        self.formation_error = None
        self.tracking_error = None
        self.control_effort = None


        # local stable test
        eigvals = np.linalg.eigvals(self.L)
        lambda_ = 1.5
        A_cl = self.A0 + lambda_ * self.B0 @ self.Gamma_1
        np.linalg.eigvals(A_cl)
        
        print("Eigenvalues of A_cl:", np.linalg.eigvals(A_cl))

    def step(self, actions):
        self.current_step += 1
        for i in range(self.num_of_bots):
            # it will be [dx,dy] from [dx1, dy1, dx2, dy2, dx3, dy3]
            self.robots[i].action = actions[2*i : 2*i+2]
        # print(f"Robot actions:")
        # print(actions.reshape(3, 2))


        self.formation_anchor = self.formation_anchor + self.FORMATION_VELOCITY
        direction = self.target - self.formation_anchor
        distance_to_target = np.linalg.norm(direction)

        self.last_action = actions

        # if distance_to_target > 5:  # only move if not yet at target
        #     # velocity = direction / distance * 40  # control speed (tune 30)
        #     velocity = direction / distance_to_target * 20  # control speed (tune 30)
        # else:
        #     velocity = np.array([0.0, 0.0])  # stop when close enough
        velocity = np.array([0.2, 0.1])
        self.FORMATION_VELOCITY = velocity

        # 3x4 matrix for desired states

        # self.desired_states = np.array([
        # [self.formation_anchor[0] + offset[0], 0, self.formation_anchor[1] + offset[1], 0]
        #     for offset in self.FORMATION_OFFSET
        # ])


        self.desired_states = np.array([
        [self.formation_anchor[0] + offset[0], self.FORMATION_VELOCITY[0], self.formation_anchor[1] + offset[1], self.FORMATION_VELOCITY[1]]
            for offset in self.FORMATION_OFFSET
        ])

        old_states = np.array([r.state.copy() for r in self.robots])
        new_states = []

        for i in range(self.num_of_bots):
            Max_ACC = 10.0
            u_i = actions[2*i:2*i+2] * Max_ACC
            dq = self.A0 @ old_states[i].reshape(4, 1) + self.B0 @ u_i.reshape(2, 1)
            next_state = old_states[i] + dq.flatten() * self.dt
            self.robots[i].state = next_state
            new_states.append(next_state)

        # self.robots = new_states
        done = False
        # formation_error = np.linalg.norm(self.L1 @ error)
        # done = formation_error < threshold
        # calling get reward to any individual is fine since they have all info about others
        reward = self.get_reward()
        
        obs = self._get_observation()
        info = {}

        terminiated = False
        truncated = (self.formation_error > 3 or self.tracking_error > 3)  # or any other condition for truncation
        return obs, reward, done, truncated, info
    
    def get_reward(self):
        w1 = 9.4
        w2 = 0.3
        w3 = 0.01
        # w1 = 4.7
        # w2 = 0.3
        # w3 = 0.01
        #dist_to_target = np.linalg.norm(self.formation_anchor - self.target)

        error = self.robots[0].get_obs(self.robots, self.desired_states)
        robot_positions = np.array([r.state[[0, 2]] for r in self.robots])
        robot_center = np.mean(robot_positions, axis=0) 

        self.formation_error = np.linalg.norm(self.L1 @ error) /1000.0
        self.tracking_error = np.linalg.norm(error) / 1000.0
        self.control_effort = np.linalg.norm(self.last_action) / 10.0 

        # if self.prev_target_dist is None:
        #     progress_reward = 0.0
        # else:
        #     progress_reward = self.prev_target_dist - dist_to_target

        # progress_reward = np.clip(progress_reward, -0.05, 0.05)
    
        # robot_velocities = np.array([r.state[[1, 3]] for r in self.robots])
        # center_velocity = np.mean(robot_velocities, axis=0)
        # center_speed = np.linalg.norm(center_velocity)

        # stop_penalty = 0.0
        # if dist_to_target < 0.3:
        #     stop_penalty = center_speed**2

        # self.prev_target_dist = dist_to_target

        # panaializing individual robot errors to see if we can get better reward design
        pos_errors = []
        # for i, r in enumerate(self.robots):
        #     pos_i = r.state[[0, 2]]
        #     desired_pos_i = self.desired_states[i][[0, 2]]
        #     pos_errors.append(np.linalg.norm(pos_i - desired_pos_i))

        #     worst_robot_error = max(pos_errors) / 1000.0
        #     avg_robot_error = np.mean(pos_errors) / 1000.0
            
        # base_reward = (-w1 * (formation_error)**2 - w2 * (tracking_error)**2
        #                 - w3 * (control_effort)**2 + 20 * progress_reward -
        #                   2 * stop_penalty     -0.5 * avg_robot_error**2
        #                     -2.0 * worst_robot_error**2)    

        
        base_reward = (-w1 * (self.formation_error)**2  - w2 * (self.tracking_error)**2 - w3 * (self.control_effort)**2)    
        
        # if dist_to_target < 0.1:
        #     base_reward += 150  # Add a bonus for being close to the target
        # elif dist_to_target < 0.3:
        #     base_reward += 50   # Add a smaller bonus for being somewhat close to the target
        # else:
        #     base_reward += 0   # No bonus if the robot is far from the target
        
        # base_reward = (-w1 * (formation_error)**2 - w2 * (tracking_error)**2
        #                 - w3 * (control_effort)**2 + 10* progress_reward) 

        #base_reward = -w1 * (formation_error)**2 - w3 * (control_effort)**2
        tracking_improve = 0.0
        formation_improve = 0.0
        if self.prev_formation_error is not None:
            formation_improve = self.prev_formation_error - self.formation_error
            tracking_improve = np.clip(tracking_improve, -0.05, 0.05)
            # print("xxx", self.prev_formation_error, self.formation_error)
            # print("formation improve: ", 3000 * formation_improve)

        if self.prev_tracking_error is not None:
            tracking_improve = self.prev_tracking_error - self.tracking_error
            tracking_improve = np.clip(tracking_improve, -0.05, 0.05)
            # print("xxx", self.prev_tracking_error, self.tracking_error)
            # print("tracking improve: ", 1000 *tracking_improve)

        self.prev_formation_error = self.formation_error
        self.prev_tracking_error = self.tracking_error
        # current_error = formation_error + tracking_error

        stable_bonus = 0.0
        if self.formation_error < 0.2:
            stable_bonus += 0.5
        if self.tracking_error < 0.5:
            stable_bonus += 0.2
        reward = base_reward + 300 * formation_improve + 100 * tracking_improve + stable_bonus
        # self.prev_error = current_error
        # reward = formation_reward + tracking_reward - w3 * (control_effort)**2

        # print("error",formation_error, control_effort)
        # print("tuned error",w1 * formation_error**2, w2 * tracking_error**2, w3 * control_effort**2)
        # print("base reward", base_reward)
        # print("tracking_error" , tracking_error**2)
        # print("  ", reward)
        # # print("Formation error ", formation_error**2)
        # # print("Worst robot error ", 2.0 * worst_robot_error**2)
        # # print("Average robot error ", 0.5 * avg_robot_error**2)

        # print("tracking reward ", tracking_improve)
        # print("formation reward: ", formation_improve)
        # print("================================")
        # reward = base_reward
        
        return reward
    
    def reset(self, seed = None, options = None):
        self.robots = []
        self.current_step = 0
        self.prev_error = 0.0

        for i in range(self.num_of_bots):
            # robot = base.Robot(i, (np.random.uniform(10, self.WIDTH - 100),   # random x within (10, 100)
            #                        np.random.uniform(10, self.HEIGHT - 100)))   # random y within bounds
            robot = base.Robot(i, (np.random.uniform(800, 900),   # random x within (10, 100)
                                   np.random.uniform(500, 600)))   # random y within bounds
            # robot = base.Robot(i, (np.random.uniform(900, 950),   # random x within (10, 100)
            #                        np.random.uniform(600, 650)))   # random y within bounds
            robot.neighbor_indexs = np.where(self.A[i] == 1)[0].tolist()  # Get indices of neighbors from adjacency matrix

            self.robots.append(robot)

        self.formation_anchor = self.robots[0].state[[0, 2]]
        
        # self.target = np.array([random.uniform(100, self.WIDTH - 100), random.uniform(100, self.HEIGHT - 100)]) 
        self.target = np.array([1040, 640]) 
        obs = self._get_observation()
        info = {}
        return obs, info

    def _get_observation(self):
        q_all = np.array([r.state for r in self.robots])
        
        self.desired_states = np.array([
        [self.formation_anchor[0] + offset[0], self.FORMATION_VELOCITY[0], self.formation_anchor[1] + offset[1], self.FORMATION_VELOCITY[1]]
            for offset in self.FORMATION_OFFSET
        ])
        
        obs = []
        for i in range(self.num_of_bots):
            own_state = self.robots[i].state
            own_error = own_state - self.desired_states[i]
            neighbor_info = []
            for j in self.robots[i].neighbor_indexs:
                neighbor_error = self.robots[j].state - self.desired_states[j]
                # Here you can compute any additional features based on neighbor states and desired states
                relative_error = own_error - neighbor_error
                neighbor_info.append(relative_error)  # Add relative error to neighbor info

            if len(neighbor_info) > 0:
                neighbor_info = np.concatenate(neighbor_info)
            else:
                neighbor_info = np.zeros(4 * len(self.robots[i].neighbor_indexs))  # No neighbors, so fill with zeros
            
            robot_obs = np.concatenate([own_state, own_error, neighbor_info])

            obs.append(robot_obs)   

        obs = np.concatenate(obs).astype(np.float32)
        # print("obs shape:", obs.shape)
        return obs
    
    def render(self):
        for r in self.robots:
            r.draw(self.screen)
        pygame.display.flip()
        
    def close(self):
        pygame.quit()