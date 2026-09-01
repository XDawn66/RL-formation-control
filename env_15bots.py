import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import Formation_control_A as base
import random


def make_ring_adjacency(n):
    A = np.zeros((n, n))

    for i in range(n):
        A[i, (i - 1) % n] = 1
        A[i, (i + 1) % n] = 1

    return A

def make_circle_offsets(n, radius=60):
    theta = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    )

    return np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta)
    ])

class FormationEnv(gym.Env):
    def __init__(self, screen):
        self.WIDTH = 1980
        self.HEIGHT = 1080
        self.screen = screen
        self.dt = 0.005

        self.robots = []
        self.num_of_bots = 15

        self.max_robots = 20
        self.robot_feature_dim = 4
        
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2* self.num_of_bots,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(48,), dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_robots, self.robot_feature_dim),  # 5 robots * (4 states + 4 errors + 4 neighbors * 4 states) = 5 * (4 + 4 + 16) = 5 * 24 = 120, but we have only 5 robots and each has 4 states and 4 errors, plus neighbor info   
            dtype=np.float32,
        )

        self.FORMATION_VELOCITY = np.array([40, 20])
        # try to make a triangle
        self.FORMATION_OFFSET = make_circle_offsets(self.num_of_bots)

        # adjacency matrix for a directed graph
        # self.A = np.array([
        #     [0, 1, 0],
        #     [0, 0, 1],
        #   
        #   [1, 0, 0]
        # ])
        
        # [a, b, c, d, e, f, g] where a,b,c,d,e,f,g are the robots and the value is 1 if they are neighbors
        self.A = make_ring_adjacency(self.num_of_bots)

        self.Degree_matrix = np.diag(
            np.sum(self.A, axis=1)
        )

        D_inv = np.linalg.inv(self.Degree_matrix)

        self.L = (
            np.eye(self.num_of_bots)
            - D_inv @ self.A
        )

        self.L1 = np.kron(
            self.L,
            np.eye(4)
        )

        #identity matrix
        self.I = np.eye(self.num_of_bots)

        # Laplacian matrix 5x5
        D_inv = np.linalg.inv(self.Degree_matrix)
        self.L = self.I - D_inv @ self.A
        print("Laplacian Matrix L:\n", self.L)

        self.I_m = np.eye(self.num_of_bots)
  
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

        # fix the formation anchor point to a specific location in the environment
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
        # self.Gamma_1 = [
        #     [-50.0, -30.0,  0.0,  0.0], #gain control feedback for x-driection
        #     [  0.0,  0.0, -50.0, -30.0] #gain control feedback for y-driection
        # ]
        self.Gamma_1 = None
        # shape (2×4), controls x and y

        #self.Gamma = np.kron(np.eye(self.num_of_bots), self.Gamma_1)  # shape (6×12)

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
        
        # for divergence detection
        self.best_convergence_error = np.inf
        self.divergence_counter = 0
        self.control_effort = 0

        self.divergence_patience = 200   # 200 steps = 1 simulated second at dt=0.005
        self.divergence_margin = 0.32    # must be 20% worse than best error
        self.warmup_steps = 200          # do not terminate immediately after reset
        self.max_episode_steps = 20000

        # local stable test
        # eigvals = np.linalg.eigvals(self.L)
        # lambda_ = 1.5
        # A_cl = self.A0 + lambda_ * self.B0 @ self.Gamma_1
        # np.linalg.eigvals(A_cl)
        
        # print("Eigenvalues of A_cl:", np.linalg.eigvals(A_cl))
        self.gamma_history = []
        self.formation_error_history = []
        self.stable_steps = 0

    def step(self, actions):

        self.current_step += 1
        velocity = np.array([40, 20])
        self.FORMATION_VELOCITY = velocity

        self.robot_positions = np.array([r.state[[0, 2]] for r in self.robots])
        self.robot_center = np.mean(self.robot_positions, axis=0)

        robot_velocities = np.array([r.state[[1, 3]] for r in self.robots])
        center_vel = np.mean(robot_velocities, axis=0)

        Kp_track = 0.3
        Kd_track = 0.5

        u_track = (
            Kp_track * (self.formation_anchor - self.robot_center)
            + Kd_track * (self.FORMATION_VELOCITY - center_vel)
        )

        for i in range(self.num_of_bots):
            # it will be [dx,dy] from [dx1, dy1, dx2, dy2, dx3, dy3]
            # self.robots[i].action = actions[2*i : 2*i+2]
            self.robots[i].action = actions
        # print(f"Robot actions:")
        # print(actions.reshape(3, 2))

        raw_g1, raw_g2 = actions

        g1 = 1.0 * (raw_g1 + 1.0) / 2.0
        g2 = 1.0 * (raw_g2 + 1.0) / 2.0
   

        self.Gamma_1 = np.array([
            [-g1, -g2,  0.0,  0.0], #gain control feedback for x-driection
            [ 0.0,  0.0, -g1, -g2] #gain control feedback for y-driection
        ])

        
        # self.Gamma_1 = np.array([
        #     [-0.043, -0.0158,  0.0,  0.0], #gain control feedback for x-driection
        #     [ 0.0,  0.0, -0.043, -0.0158] #gain control feedback for y-driection
        # ])

        self.Gamma = np.kron(np.eye(self.num_of_bots), self.Gamma_1)  # shape (6×12)

        error = np.array([r.state for r in self.robots]).flatten() - self.desired_states.flatten()

        rho = self.L1 @ error

        r = self.Gamma @ rho

        r = np.clip(r, -10.0, 10.0)

        #print(f"self.Gamma_1: {self.Gamma_1}")
        
        for i, robot in enumerate(self.robots):
            r_i = r[2*i:2*i+2]
            r_i += u_track

            dq = self.A0 @ robot.state.reshape(4, 1) + self.B0 @ r_i.reshape(2, 1)
            robot.state += dq.flatten() * self.dt
            # print(f"dq for robot {i}: {dq.flatten() * self.dt}")

        # r_i += FORMATION_VELOCITY
        # dq = A0 @ self.state.reshape(4, 1) + B0 @ r_i.reshape(2, 1)
        # self.state += dq.flatten() * DT


        self.formation_anchor += self.FORMATION_VELOCITY * self.dt
        # self.formation_anchor = [0,0]
        # direction = self.target - self.formation_anchor
        # distance_to_target = np.linalg.norm(direction)

        self.last_action = actions


        self.desired_states = np.array([
        [self.formation_anchor[0] + offset[0], self.FORMATION_VELOCITY[0], self.formation_anchor[1] + offset[1], self.FORMATION_VELOCITY[1]]
            for offset in self.FORMATION_OFFSET
        ])

        old_states = np.array([r.state.copy() for r in self.robots])
        new_states = []

        # uniform control for direct rl control
        # for i in range(self.num_of_bots):
        #     Max_ACC = 10.0
        #     u_i = actions[2*i:2*i+2] * Max_ACC
        #     dq = self.A0 @ old_states[i].reshape(4, 1) + self.B0 @ u_i.reshape(2, 1)
        #     next_state = old_states[i] + dq.flatten() * self.dt
        #     self.robots[i].state = next_state
        #     new_states.append(next_state)

        # self.robots = new_states
        done = False
        # formation_error = np.linalg.norm(self.L1 @ error)
        # done = formation_error < threshold
        # calling get reward to any individual is fine since they have all info about others
        reward = self.get_reward()
        reward = float(np.clip(reward, -10.0, 10.0))

        self.gamma_history.append([g1, g2])
        self.formation_error_history.append(self.formation_error)  

        failure = False

        if self.current_step > self.warmup_steps:

            # Update best result seen during this episode
            if self.convergence_error < self.best_convergence_error:
                self.best_convergence_error = self.convergence_error
                self.divergence_counter = 0

            # Significantly worse than the best state reached so far
            elif self.convergence_error > (
                self.best_convergence_error * (1.0 + self.divergence_margin)
            ):
                self.divergence_counter += 1

            # Slightly worse, but not enough to call it divergence
            else:
                self.divergence_counter = max(0, self.divergence_counter - 1)

            if self.divergence_counter >= self.divergence_patience:
                failure = True
        
        obs = self._get_observation()
        info = {}

        #terminated = failure
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        if failure:
            reward = -10.0

        info = {
            "convergence_error": float(self.convergence_error),
            "best_convergence_error": float(self.best_convergence_error),
            "divergence_counter": self.divergence_counter,
            "failure": failure,
        }

        if self.current_step % 1000 == 0:
            print(
                "step:", self.current_step,
                "raw action:", actions,
                "g1:", g1,
                "g2:", g2
            )
            print("formation_error", self.formation_error, "tracking_error", self.tracking_error)

        if self.current_step % 1000 == 0:
            print("================================")
            print("step:", self.current_step)

            print(
                "center-anchor distance:",
                np.linalg.norm(
                    self.robot_center - np.mean(
                        np.array([s[[0, 2]] for s in self.desired_states]),
                        axis=0
                    )
                )
            )

            desired_positions = np.array([
                s[[0, 2]]
                for s in self.desired_states
            ])

            desired_center = np.mean(
                desired_positions,
                axis=0
            )

            center_tracking_error = np.linalg.norm(
                self.robot_center - desired_center
            )

            print(
                "u_track magnitude:",
                np.linalg.norm(u_track)
            )

            print(
                "consensus magnitude:",
                np.linalg.norm(r)
            )

            print(
                "formation error:",
                self.formation_error
            )

            print(
                "tracking error:",
                self.tracking_error
            )

            print("g1:", g1, "g2:", g2)

        if self.current_step % 1000 == 0:

            print("Individual position errors:")

            for i, robot in enumerate(self.robots):
                pos = robot.state[[0, 2]]
                desired_pos = self.desired_states[i][[0, 2]]

                print(
                    i,
                    np.linalg.norm(pos - desired_pos)
                )

            print(
                "desired-center tracking:",
                center_tracking_error
            )

            print(
                "consensus effort:",
                np.linalg.norm(r)
            )

                    

        return obs, reward, terminated, truncated, info
    
    def get_reward(self):
        w1 = 9.4
        w2 = 6.0
        w3 = 0.01
        # w1 = 4.7
        # w2 = 0.3
        # w3 = 0.01
        #dist_to_target = np.linalg.norm(self.formation_anchor - self.target)


        desired_positions = np.array([s[[0, 2]] for s in self.desired_states])
        desired_center = np.mean(desired_positions, axis=0)
        error = self.robots[0].get_obs(self.robots, self.desired_states)
        self.robot_positions = np.array([r.state[[0, 2]] for r in self.robots])
        self.robot_center = np.mean(self.robot_positions, axis=0)

        self.formation_error = np.linalg.norm(self.L1 @ error) /1000.0
        self.tracking_error = np.linalg.norm(error) / 1000.0
        self.control_effort = np.linalg.norm(self.last_action) / 10.0 


        # self.prev_target_dist = dist_to_target

        # panaializing individual robot errors to see if we can get better reward design
        pos_errors = []

        
        base_reward = (-w1 * (self.formation_error)**2  - w2 * (self.tracking_error)**2 - w3 * (self.control_effort)**2)    
        
        tracking_improve = 0.0
 
        self.prev_formation_error = self.formation_error
        self.prev_tracking_error = self.tracking_error
        # current_error = formation_error + tracking_error

        # stable_bonus = 0.0
        # if self.formation_error < 0.2:
        #     stable_bonus += 0.5
        # if self.tracking_error < 0.5:
        #     stable_bonus += 0.2

        stable_bonus = np.exp(-4 * self.formation_error)

        pos_errors = []
        for i, r in enumerate(self.robots):
            pos_i = r.state[[0, 2]]
            desired_i = self.desired_states[i][[0, 2]]
            pos_errors.append(np.linalg.norm(pos_i - desired_i))

        avg_robot_error = np.mean(pos_errors) / 1000.0
        worst_robot_error = np.max(pos_errors) / 1000.0

        self.convergence_error = (avg_robot_error + 2.0 * worst_robot_error)

        reward = base_reward + stable_bonus -2.0 * avg_robot_error**2 -5.0 * worst_robot_error**2

        good_formation = (
            self.formation_error < 0.8
            and self.tracking_error < 0.8
        )

        if good_formation:
            self.stable_steps += 1
        else:
            self.stable_steps = 0

        stay_bonus = (
    self.stable_steps / 500.0
        ) * max(0.0, 1.0 - self.formation_error)

        reward += min(stay_bonus, 1.0)

        reward += stay_bonus
        # self.prev_error = current_error
        # reward = formation_reward + tracking_reward - w3 * (control_effort)**2

        # print("error",formation_error, control_effort)
        # print("tuned error",w1 * formation_error**2, w2 * tracking_error**2, w3 * control_effort**2)
        # print("base reward", base_reward)
        # print("tracking_error" , tracking_error**2)
        # print("  ", reward)
        # print("spread penalty: ", spread_penalty)
        # print("worst bot penalty: ", worst_bot_penalty)
        # # print("Formation error ", formation_error**2)
        #print("Worst robot error ", 2.0 * worst_robot_error**2)
        #print("Average robot error ", 0.5 * avg_robot_error**2)

        # print("tracking reward ", tracking_improve)
        # print("formation reward: ", formation_improve)
        # print("================================")
        # reward = base_reward
        
        return reward
    
    def reset(self, seed = None, options = None):
        self.robots = []
        self.current_step = 0
        self.prev_error = 0.0
        self.best_convergence_error = np.inf
        self.divergence_counter = 0

        self.prev_formation_error = None
        self.prev_tracking_error = None

        for i in range(self.num_of_bots):
            # robot = base.Robot(i, (np.random.uniform(10, self.WIDTH - 100),   # random x within (10, 100)
            #                        np.random.uniform(10, self.HEIGHT - 100)))   # random y within bounds
            robot = base.Robot(i, (np.random.uniform(400, 1100),   # random x within (10, 100)
                                   np.random.uniform(200, 900)))   # random y within bounds
            # robot = base.Robot(i, (np.random.uniform(900, 950),   # random x within (10, 100)
            #                        np.random.uniform(600, 650)))   # random y within bounds
            robot.neighbor_indexs = np.where(self.A[i] == 1)[0].tolist()  # Get indices of neighbors from adjacency matrix
            # print(f"Robot {i} neighbors: {robot.neighbor_indexs}")
            self.robots.append(robot)

        self.formation_anchor = self.robots[0].state[[0, 2]]

        self.gamma_history.clear()
        self.formation_error_history.clear()
        
        # self.target = np.array([random.uniform(100, self.WIDTH - 100), random.uniform(100, self.HEIGHT - 100)]) 
        self.target = np.array([1040, 640]) 
        obs = self._get_observation()
        info = {}
        return obs, info

    def _get_observation(self):
        
        self.desired_states = np.array([
        [self.formation_anchor[0] + offset[0], self.FORMATION_VELOCITY[0], self.formation_anchor[1] + offset[1], self.FORMATION_VELOCITY[1]]
            for offset in self.FORMATION_OFFSET
        ])
        
        
        obs = np.zeros(
            (self.max_robots, self.robot_feature_dim),
            dtype=np.float32
        )

        for i in range(self.num_of_bots):
            own_state = self.robots[i].state
            own_error = own_state - self.desired_states[i]

            x_normalize = 100.0
            y_normalize = 50.0

            obs[i] = np.array([
            own_error[0] / x_normalize,  # x position error
            own_error[1] / y_normalize,   # x velocity error
            own_error[2] / x_normalize,  # y position error
            own_error[3] / y_normalize    # y velocity error
            ], dtype=np.float32)
            # if len(neighbor_info) > 0:
            #     neighbor_info = np.concatenate(neighbor_info)
            # else:
            #     neighbor_info = np.zeros(4 * len(self.robots[i].neighbor_indexs))  # No neighbors, so fill with zeros
            
            # robot_obs = np.concatenate([own_state, own_error, neighbor_info])

            # obs.append(robot_token)

        # obs = np.concatenate(obs).astype(np.float32)
        # print("obs shape:", obs.shape)
        return obs
    
    def render(self):
        for r in self.robots:
            r.draw(self.screen)
        pygame.display.flip()
        
    def close(self):
        pygame.quit()
