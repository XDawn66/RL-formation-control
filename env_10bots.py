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
        self.num_of_bots = 10

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

        self.formation_error = 999
        self.tracking_error = None
        self.control_effort = None
        
        # for divergence detection
        self.best_convergence_error = np.inf
        self.divergence_counter = 0
        self.control_effort = 0

        self.divergence_patience = 200   # 200 steps = 1 simulated second at dt=0.005
        self.divergence_margin = 0.32    # must be 20% worse than best error
        self.warmup_steps = 200          # do not terminate immediately after reset
        self.max_episode_steps = 6000

        # local stable test
        # eigvals = np.linalg.eigvals(self.L)
        # lambda_ = 1.5
        # A_cl = self.A0 + lambda_ * self.B0 @ self.Gamma_1
        # np.linalg.eigvals(A_cl)
        
        # print("Eigenvalues of A_cl:", np.linalg.eigvals(A_cl))
        self.gamma_history = []
        self.formation_error_history = []
        self.stable_steps = 0

                  # for avoidance mode
        self.safe_distance = 30.0  # safe distance for avoidance mode
        self.robot_radius = 5.0  # radius of the robot for avoidance mode
        self.sensing_radius = 500.0
        self.avoidance_distance = 100.0

        self.obstacle_spawned = False
        self.obstacles = []
        self.obstacles_radius = 15.0

        self.obstacle_spawn_count = 0
        self.max_obstacle_spawns = 10

        self.last_obstacle_spawn_step = -999999
        self.obstacle_spawn_cooldown = 12000

        self.FORMATION_READY_THRESHOLD = 0.03
        self.OBSTACLE_SPAWN_DISTANCE = 1500.0

        self.current_wall_dir = None
        self.current_wall_angle = None
        self.current_wall_half_length = None

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

         # Phase 1: let every robot run Rule 1 / Rule 2
        for i in range(self.num_of_bots):
            self.desired_states[i] = self.calculate_avoidance_subgoal(i)

        # Phase 2: after everyone has chosen, fix tiny 1-robot groups
        self.rebalance_avoidance_groups()

        # Phase 3: recompute avoidance subgoals if any side changed
        for i in range(self.num_of_bots):
            if self.robots[i].mode == 1:
                self.desired_states[i] = self.calculate_avoidance_subgoal(i)

        error = np.array([r.state for r in self.robots]).flatten() - self.desired_states.flatten()

        rho = self.L1 @ error

        r = self.Gamma @ rho

        r = np.clip(r, -10.0, 10.0)

        
    # =========================================
    # SPAWN OBSTACLE AFTER FORMATION CONVERGES
    # =========================================

        can_spawn_again = (
            self.current_step - self.last_obstacle_spawn_step
            >= self.obstacle_spawn_cooldown
        )

        if (
            self.formation_error < self.FORMATION_READY_THRESHOLD
            and can_spawn_again
            and self.obstacle_spawn_count < self.max_obstacle_spawns
        ):
            self.spawn_test_wall()
        
        for i, robot in enumerate(self.robots):
                r_i = r[2*i:2*i+2].copy()
    
                if robot.mode == 1:
    
                    # Keep some formation coupling
                    # r_i *= 0.25
                    r_i *= 0.05
    
                    pos = robot.state[[0, 2]]
                    vel = robot.state[[1, 3]]
    
                    desired_pos = self.desired_states[i][[0, 2]]
    
                    # During avoidance, don't demand [40,20] immediately.
                    # Let position control generate the maneuver.
                    desired_vel = 0.4 * self.FORMATION_VELOCITY
    
                    Kp_avoid = 2.0
                    Kd_avoid = 1.3
    
                    u_subgoal = (
                        Kp_avoid * (desired_pos - pos)
                        + Kd_avoid * (desired_vel - vel)
                    )
    
                    # speed reugaltion
                    travel_dir = (
                        self.FORMATION_VELOCITY.astype(float)
                        / np.linalg.norm(self.FORMATION_VELOCITY)
                    )
    
                    forward_speed = np.dot(vel, travel_dir)
    
                    desired_forward_speed = np.dot(
                        desired_vel,
                        travel_dir
                    )
    
                    forward_acc = np.dot(
                        u_subgoal,
                        travel_dir
                    )
    
                    # Already too fast:
                    # don't allow positive acceleration along travel direction
                    if (
                        forward_speed > desired_forward_speed
                        and forward_acc > 0
                    ):
                        u_subgoal -= forward_acc * travel_dir
    
                    # Give avoidance real authority
                    u_subgoal = np.clip(
                        u_subgoal,
                        -10.0,
                        10.0
                    )
    
                    consensus_raw = r_i.copy()
    
                    r_i += u_subgoal
    
                    # only tiny global tracking while avoiding
                    r_i += 0.05 * u_track
    
                else:
                    r_i += u_track

                r_i = np.clip(
                    r_i,
                    -10.0,
                    10.0
                )

                dq = self.A0 @ robot.state.reshape(4, 1) + self.B0 @ r_i.reshape(2, 1)
                robot.state += dq.flatten() * self.dt

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

        self.obstacles = []
        self.obstacle_spawned = False
        
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
    
    def render(self, camera_x=0.0, camera_y=0.0):
        for r in self.robots:
            r.draw(self.screen, camera_x, camera_y)

        pygame.display.flip()
        
    def close(self):
        pygame.quit()

    def spawn_test_wall(self):

        travel_dir = (
            self.FORMATION_VELOCITY.astype(float)
            / np.linalg.norm(self.FORMATION_VELOCITY)
        )

        # -----------------------------
        # RANDOM WALL PARAMETERS
        # -----------------------------

        half_length = np.random.uniform(60.0, 180.0)

        angle_offset_deg = np.random.uniform(-35.0, 35.0)
        angle_offset = np.deg2rad(angle_offset_deg)

        num_points = np.random.randint(7, 16)

        # -----------------------------
        # BASE WALL DIRECTION
        # perpendicular to travel
        # -----------------------------

        base_wall_dir = np.array([
            -travel_dir[1],
            travel_dir[0]
        ])

        # -----------------------------
        # ROTATE WALL
        # -----------------------------

        c = np.cos(angle_offset)
        s = np.sin(angle_offset)

        R = np.array([
            [c, -s],
            [s,  c]
        ])

        wall_dir = R @ base_wall_dir

        self.current_wall_dir = wall_dir.copy()
        self.current_wall_angle = angle_offset_deg
        self.current_wall_half_length = half_length

        # -----------------------------
        # PLACE WALL AHEAD
        # -----------------------------

        wall_center = (
            self.formation_anchor
            + self.OBSTACLE_SPAWN_DISTANCE * travel_dir
        )

        # remove previous wall
        self.obstacles = []

        for offset in np.linspace(
            -half_length,
            half_length,
            num_points
        ):
            point = wall_center + offset * wall_dir
            self.obstacles.append(point)

        self.obstacle_spawn_count += 1
        self.last_obstacle_spawn_step = self.current_step


    def calculate_avoidance_subgoal(self, robot_idx):
        robot = self.robots[robot_idx]
        own_state = self.robots[robot_idx].state
        own_pos = own_state[[0, 2]]

        nearest_obstacle = None
        min_distance = float("inf")

        # -----------------------------------
        # Geometry of current wall
        # -----------------------------------

        wall_points = np.array(self.obstacles, dtype=float)

        if len(wall_points) == 0:
            self.robots[robot_idx].mode = 0
            return self.desired_states[robot_idx]

        if self.current_wall_dir is None:
                for i, robot in enumerate(self.robots):
                    robot.mode = 0
                return self.desired_states[robot_idx]

        wall_center = np.mean(wall_points, axis=0)

        travel_dir = self.FORMATION_VELOCITY.astype(float)
        travel_dir /= np.linalg.norm(travel_dir)

        # wall runs perpendicular to travel
        wall_dir = self.current_wall_dir

        # project wall points onto wall direction
        wall_projection = (
            (wall_points - wall_center) @ wall_dir
        )

        end_1 = (
            wall_center
            + np.min(wall_projection) * wall_dir
        )

        end_2 = (
            wall_center
            + np.max(wall_projection) * wall_dir
        )

        margin = 140.0
        forward_margin = 120.0

        exit_1 = (
            end_1
            - margin * wall_dir
            + forward_margin * travel_dir
        )

        exit_2 = (
            end_2
            + margin * wall_dir
            + forward_margin * travel_dir
        )

        exit_1 = force_exit_forward(
            exit_1,
            wall_center,
            travel_dir,
            min_forward=150.0
        )

        exit_2 = force_exit_forward(
            exit_2,
            wall_center,
            travel_dir,
            min_forward=150.0
        )

        # Find nearest sensed obstacle
        for obstacle in self.obstacles:
            obs_pos = np.asarray(obstacle, dtype=float)

            distance = np.linalg.norm(own_pos - obs_pos)

            rel_pos = obs_pos - own_pos

            # only consider obstacle points ahead
            if np.dot(rel_pos, travel_dir) <= 0:
                continue

            if (
                distance < min_distance
                and distance < self.sensing_radius
            ):
                min_distance = distance
                nearest_obstacle = obstacle

            if nearest_obstacle is None:
                robot.mode = 0
                robot.avoid_side = None
                return self.desired_states[robot_idx]

            obs_pos = np.asarray(nearest_obstacle, dtype=float)
            rel_pos = obs_pos - own_pos

            if np.dot(rel_pos, travel_dir) <= 0:
                continue

            if distance < min_distance and distance < self.sensing_radius:
                min_distance = distance
                nearest_obstacle = obstacle



        # -----------------------------------
        # MODE SWITCHING / HYSTERESIS
        # -----------------------------------

        # No obstacle sensed at all
        if nearest_obstacle is None:
            robot.mode = 0
            robot.avoid_side = None
            return self.desired_states[robot_idx]

        rel_pos = obs_pos - own_pos
        dist = np.linalg.norm(rel_pos)

        obs_dir = rel_pos / (dist + 1e-6)

        robot_vel = own_state[[1, 3]]

        closing_speed = np.dot(robot_vel, obs_dir)

        base_trigger = 380.0
        speed_gain = 15.0

        trigger_distance = (
            base_trigger
            + speed_gain * max(closing_speed, 0.0)
        )

        if robot.mode == 0:
            if min_distance < trigger_distance:
                robot.mode = 1

        elif robot.mode == 1:

            # how far robot has traveled past the wall
            wall_progress = np.dot(
                own_pos - wall_center,
                travel_dir
            )

            PASS_MARGIN = 160.0

            # robot has cleared the wall -> stop avoidance
            if wall_progress > PASS_MARGIN:
                robot.mode = 0
                robot.avoid_side = None
                return self.desired_states[robot_idx]

        if robot.mode == 0:
            if min_distance < trigger_distance:
                robot.mode = 1

        
       # -----------------------------------
        # AVOIDANCE MODE
        # -----------------------------------

        wall_x_mean = np.mean([obs[0] for obs in self.obstacles])
        wall_y_min = min(obs[1] for obs in self.obstacles)
        wall_y_max = max(obs[1] for obs in self.obstacles)
        wall_center_y = 0.5 * (wall_y_min + wall_y_max)

      # Preserve existing side
        if (
            robot.mode == 1
            and getattr(robot, "avoid_side", None) is not None
        ):
            chosen_side = robot.avoid_side
        else:
            chosen_side = None

        cost_1 = np.linalg.norm(exit_1 - own_pos)
        cost_2 = np.linalg.norm(exit_2 - own_pos)

        own_side = -1.0 if cost_1 < cost_2 else 1.0

        cost_difference = abs(cost_1 - cost_2)

        GEOMETRY_THRESHOLD = 80.0

        if cost_difference > GEOMETRY_THRESHOLD:
            chosen_side = own_side

        # -------------------------
        # RULE 1
        # ONLY if robot has no side yet
        # -------------------------
        if chosen_side is None:

            candidate_neighbor = None
            best_progress = -float("inf")

            for neighbor_idx in robot.neighbor_indexs:

                neighbor = self.robots[neighbor_idx]

                if (
                    neighbor.mode == 1
                    and getattr(neighbor, "avoid_side", None) is not None
                ):

                    neighbor_pos = neighbor.state[[0, 2]]

                    rel_to_anchor = (
                        neighbor_pos - self.formation_anchor
                    )

                    progress = np.dot(
                        rel_to_anchor,
                        travel_dir
                    )

                    if progress > best_progress:
                        best_progress = progress
                        candidate_neighbor = neighbor

            if candidate_neighbor is not None:
                chosen_side = candidate_neighbor.avoid_side


        # -------------------------
        # RULE 2: own geometry choice
        # only if Rule 1 gave no direction
        # -------------------------
        if chosen_side is None:

            exit_1 = (
                end_1
                - margin * wall_dir
                + forward_margin * travel_dir
            )

            exit_2 = (
                end_2
                + margin * wall_dir
                + forward_margin * travel_dir
            )

            exit_1 = force_exit_forward(
                exit_1,
                wall_center,
                travel_dir,
                min_forward=150.0
            )

            exit_2 = force_exit_forward(
                exit_2,
                wall_center,
                travel_dir,
                min_forward=150.0
            )

            # print(
            #     "exit_1:", exit_1,
            #     "exit_2:", exit_2
            # )

            cost_1 = np.linalg.norm(exit_1 - own_pos)
            cost_2 = np.linalg.norm(exit_2 - own_pos)

            if cost_1 < cost_2:
                chosen_side = -1.0   # UP
            else:
                chosen_side = 1.0    # DOWN


        robot.avoid_side = chosen_side


        # -------------------------
        # actual subgoal
        # -------------------------
        margin = 60.0

        if robot.avoid_side < 0:
            target_pos = exit_1
        else:
            target_pos = exit_2

        subgoal = self.desired_states[robot_idx].copy()

        subgoal[0] = target_pos[0]
        subgoal[2] = target_pos[1]

        side_dir = np.array([
                    -travel_dir[1],
                    travel_dir[0]
                ])

        return np.array([
            subgoal[0],
            0.0,             # important
            subgoal[2],
            0.0
        ])
    
    def rebalance_avoidance_groups(self):

        active = [
            i for i, r in enumerate(self.robots)
            if r.mode == 1 and getattr(r, "avoid_side", None) is not None
        ]

        if len(active) < 3:
            return

        side_neg = [
            i for i in active
            if self.robots[i].avoid_side < 0
        ]

        side_pos = [
            i for i in active
            if self.robots[i].avoid_side > 0
        ]

        MIN_GROUP_SIZE = 2

        # no actual split, so don't force one
        if len(side_neg) == 0 or len(side_pos) == 0:
            return

        wall_points = np.asarray(self.obstacles, dtype=float)
        if len(wall_points) == 0:
            return

        wall_center = np.mean(wall_points, axis=0)

        travel_dir = self.FORMATION_VELOCITY.astype(float)
        travel_dir /= np.linalg.norm(travel_dir)

        wall_dir = self.current_wall_dir

        wall_projection = (
            wall_points - wall_center
        ) @ wall_dir

        end_1 = (
            wall_center
            + np.min(wall_projection) * wall_dir
        )

        end_2 = (
            wall_center
            + np.max(wall_projection) * wall_dir
        )

        margin = 140.0
        forward_margin = 120.0

        exit_1 = (
            end_1
            - margin * wall_dir
            + forward_margin * travel_dir
        )

        exit_2 = (
            end_2
            + margin * wall_dir
            + forward_margin * travel_dir
        )

        # Case: only one robot on -1 side
        if len(side_neg) < MIN_GROUP_SIZE and len(side_pos) > MIN_GROUP_SIZE:

            candidate = min(
                side_pos,
                key=lambda i: np.linalg.norm(
                    self.robots[i].state[[0, 2]] - exit_1
                )
            )

            self.robots[candidate].avoid_side = -1.0

        # Case: only one robot on +1 side
        elif len(side_pos) < MIN_GROUP_SIZE and len(side_neg) > MIN_GROUP_SIZE:

            candidate = min(
                side_neg,
                key=lambda i: np.linalg.norm(
                    self.robots[i].state[[0, 2]] - exit_2
                )
            )

            self.robots[candidate].avoid_side = 1.0

def force_exit_forward(exit_point, wall_center, travel_dir, min_forward=120.0):

    forward_progress = np.dot(
        exit_point - wall_center,
        travel_dir
    )

    if forward_progress < min_forward:
        exit_point = (
            exit_point
            + (min_forward - forward_progress) * travel_dir
        )

    return exit_point