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

        self.robots = []
        self.num_of_bots = 5

        self.max_robots = 20
        self.robot_feature_dim = 10
        
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2* self.num_of_bots,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(48,), dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 + 2 * self.num_of_bots,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_robots, self.robot_feature_dim),  # 5 robots * (4 states + 4 errors + 4 neighbors * 4 states) = 5 * (4 + 4 + 16) = 5 * 24 = 120, but we have only 5 robots and each has 4 states and 4 errors, plus neighbor info   
            dtype=np.float32,
        )

        self.FORMATION_VELOCITY = np.array([40, 20])
        # try to make a triangle
        self.FORMATION_OFFSET = np.array([
            [0, 0],
            [30, 0],
            [0, 30],
            [50, 30],
            [16, 50]
        ])
        # adjacency matrix for a directed graph
        # self.A = np.array([
        #     [0, 1, 0],
        #     [0, 0, 1],
        #   
        #   [1, 0, 0]
        # ])
        
        # [a, b, c, d, e] where a,b,c,d,e are the robots and the value is 1 if they are neighbors
        self.A = np.array([
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0]
            ])

        self.Degree_matrix = np.array([
            [2,0,0,0,0],
            [0,2,0,0,0],
            [0,0,2,0,0],
            [0,0,0,2,0],
            [0,0,0,0,2]
        ])

        #identity matrix
        self.I = np.eye(5)

        # Laplacian matrix 5x5
        D_inv = np.linalg.inv(self.Degree_matrix)
        self.L = self.I - D_inv @ self.A
        print("Laplacian Matrix L:\n", self.L)

        self.I_m = np.eye(5)
  
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
        self.max_episode_steps = 30000

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

        self.obstacle_spawned = False
        self.obstacles = []
        self.obstacles_radius = 15.0

        self.obstacle_spawn_count = 0
        self.max_obstacle_spawns = 10

        self.last_obstacle_spawn_step = -999999
        self.obstacle_spawn_cooldown = 12000

        self.FORMATION_READY_THRESHOLD = 0.1
        self.OBSTACLE_SPAWN_DISTANCE = 1500.0


        self.avoidance_distance = 300.0
        self.robot_avoidance_distance = 60.0

        self.robot_sensing_radius = 100.0

        self.nearest_robot_dist = np.full(self.num_of_bots,self.robot_sensing_radius)

        self.nearest_obstacle_dist = np.full(
            self.num_of_bots,
            self.sensing_radius
        )

        # fixed obstacle parameters
        self.fixed_obstacle_radius = 70.0
        self.fixed_obstacle_offset = 700.0

        self.formation_stable_steps = 0
        self.required_stable_steps = 500


    def step(self, actions):

        self.current_step += 1
        velocity = np.array([40, 20])
        self.FORMATION_VELOCITY = velocity

        self.robot_positions = np.array([r.state[[0, 2]] for r in self.robots])
        self.robot_center = np.mean(self.robot_positions, axis=0)

        robot_velocities = np.array([r.state[[1, 3]] for r in self.robots])
        center_vel = np.mean(robot_velocities, axis=0)

            
        self.desired_states = np.array([
        [self.formation_anchor[0] + offset[0], self.FORMATION_VELOCITY[0], self.formation_anchor[1] + offset[1], self.FORMATION_VELOCITY[1]]
            for offset in self.FORMATION_OFFSET
        ])

        raw_g1, raw_g2 = actions[:2]
        avoid_actions = actions[2:].reshape(self.num_of_bots,2)
        
        Kp_track = 0.3
        Kd_track = 0.5

        u_track = (
            Kp_track * (self.formation_anchor - self.robot_center)
            + Kd_track * (self.FORMATION_VELOCITY - center_vel)
        )

        for i in range(self.num_of_bots):
            # it will be [dx,dy] from [dx1, dy1, dx2, dy2, dx3, dy3]
            # self.robots[i].action = actions[2*i : 2*i+2]
            # self.robots[i].action = actions

            # each robot stores avoid_x, avoid_y 
            self.robots[i].action = avoid_actions[i]
        # print(f"Robot actions:")
        # print(actions.reshape(3, 2))



        g1 = 1.0 * (raw_g1 + 1.0) / 2.0
        g2 = 1.0 * (raw_g2 + 1.0) / 2.0
   

        self.Gamma_1 = np.array([
            [-g1, -g2,  0.0,  0.0], #gain control feedback for x-driection
            [ 0.0,  0.0, -g1, -g2] #gain control feedback for y-driection
        ])

        self.Gamma = np.kron(np.eye(self.num_of_bots), self.Gamma_1)  # shape (6×12)

        normal_desired_states = np.array([
        [
            self.formation_anchor[0] + offset[0],
            self.FORMATION_VELOCITY[0],
            self.formation_anchor[1] + offset[1],
            self.FORMATION_VELOCITY[1]
        ] for offset in self.FORMATION_OFFSET])

        self.desired_states = normal_desired_states.copy()

        error = np.array([r.state for r in self.robots]).flatten() - self.desired_states.flatten()

        rho = self.L1 @ error

        r = self.Gamma @ rho

        r = np.clip(r, -10.0, 10.0)

        #print(f"self.Gamma_1: {self.Gamma_1}")


    # =========================================
    # SPAWN OBSTACLE AFTER FORMATION CONVERGES
    # =========================================

        can_spawn_again = (
            self.current_step - self.last_obstacle_spawn_step
            >= self.obstacle_spawn_cooldown
        )

        if self.formation_error < self.FORMATION_READY_THRESHOLD:
            self.formation_stable_steps += 1
        else:
            self.formation_stable_steps = 0

        if (
            self.formation_stable_steps >= self.required_stable_steps
            and can_spawn_again
            and self.obstacle_spawn_count < self.max_obstacle_spawns
        ):
            self.spawn_fixed_obstacle()

            self.formation_stable_steps = 0
                
        for i, robot in enumerate(self.robots):
            r_i = r[2*i:2*i+2].copy()

            AVOID_SCALE = 5.0

            formation_scale = 1.0

            u_avoid = np.zeros(2)

            obstacle_close = self.nearest_obstacle_dist[i] < self.avoidance_distance
            robot_close = self.nearest_robot_dist[i] < self.robot_avoidance_distance

            if obstacle_close or robot_close:
                formation_scale = 0.2
                u_avoid = AVOID_SCALE * avoid_actions[i]


            # if obstacle_close:
            #     formation_scale = 0.2
            # elif robot_close:
            #     formation_scale = 0.5

            r_i = formation_scale * r[2*i:2*i+2].copy()
            r_i += u_avoid
            r_i += u_track

            # # avoidance control
            # r_i += u_avoid

            # # tracking control
            # r_i += u_track

            # final actuator limit
            r_i = np.clip(r_i, -10.0, 10.0)

            dq = (
                self.A0 @ robot.state.reshape(4, 1)
                + self.B0 @ r_i.reshape(2, 1)
            )

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


        # self.desired_states = np.array([
        # [self.formation_anchor[0] + offset[0], self.FORMATION_VELOCITY[0], self.formation_anchor[1] + offset[1], self.FORMATION_VELOCITY[1]]
        #     for offset in self.FORMATION_OFFSET
        # ])


        reward = self.get_reward()
        reward = float(np.clip(reward, -10.0, 10.0))

        self.gamma_history.append([g1, g2])
        self.formation_error_history.append(self.formation_error)  

        failure = False

        # print(
        #     f"robot {i}: "
        #     f"formation={r[2*i:2*i+2]}, "
        #     f"avoid={u_avoid}, "
        #     f"obs_dist={self.nearest_obstacle_dist[i]}, "
        #     f"robot_dist={self.nearest_robot_dist[i]}"
        # )


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
        
        # if self.collision_penalty > 3:
        #     terminated = True
        #     reward = -10.0

        if self.robot_collision_penalty > 0:
            # terminated = True
            reward = -10.0

        if reward < -100.0:
            terminated = True
        truncated = self.current_step >= self.max_episode_steps

        if failure:
            reward = -10.0

        info = {
            "convergence_error": float(self.convergence_error),
            "best_convergence_error": float(self.best_convergence_error),
            "divergence_counter": self.divergence_counter,
            "failure": failure,
        }

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
        self.control_effort = np.mean(np.square(self.last_action))


        # self.prev_target_dist = dist_to_target

        # panaializing individual robot errors to see if we can get better reward design
        pos_errors = []

        
        base_reward = (-w1 * (self.formation_error)**2  - w2 * (self.tracking_error)**2 - w3 * (self.control_effort)**2)    
         
        self.prev_formation_error = self.formation_error
        self.prev_tracking_error = self.tracking_error

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

        stay_bonus = (self.stable_steps / 500.0) * max(0.0, 1.0 - self.formation_error)

        reward += min(stay_bonus, 1.0)

        reward += stay_bonus

        self.collision_penalty = 0.0
        self.robot_collision_penalty = 0.0
        self.near_obstacle_penalty = 0.0

        collision_distance = self.robot_radius + self.obstacles_radius

        # reward for robot to objects
        for robot in self.robots:
            robot_pos = robot.state[[0,2]]

            if len(self.obstacles) > 0 :
                obstacle_array = np.asanyarray(self.obstacles, dtype=float)

                distance = np.linalg.norm(obstacle_array - robot_pos, axis= 1)

                min_dist = np.min(distance)

                if min_dist < collision_distance:
                    self.collision_penalty += 1.0

                elif min_dist < self.safe_distance:
                    self.near_obstacle_penalty += (self.safe_distance - min_dist) / self.safe_distance

        # reward for robot to robot
        for i in range(self.num_of_bots):
            for j in range(i + 1, self.num_of_bots):

                pos_i = self.robots[i].state[[0, 2]]
                pos_j = self.robots[j].state[[0, 2]]

                dist = np.linalg.norm(pos_i - pos_j)

                collision_dist = 2.0 * self.robot_radius

                if dist < collision_dist:
                    self.robot_collision_penalty += 0.0

        reward -= 10.0 * self.collision_penalty
        reward -= 5.0 * self.robot_collision_penalty
        reward -= 2.0 * self.near_obstacle_penalty

        # print("reward", reward)
        # print("collision_penalty", self.collision_penalty)
        # print("robot_collision_penalty", self.robot_collision_penalty)
        # print("near_obstacle_penalty", self.near_obstacle_penalty)

        return reward
    
    def reset(self, seed = None, options = None):
        self.robots = []
        self.current_step = 0
        self.prev_error = 0.0
        self.best_convergence_error = np.inf
        self.divergence_counter = 0

        self.prev_formation_error = None
        self.prev_tracking_error = None

        self.obstacles = []
        self.obstacle_spawned = False
        

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
    
        
        obs = np.zeros(
            (self.max_robots, self.robot_feature_dim),
            dtype=np.float32
        )

        for i in range(self.num_of_bots):
            if self.desired_states is None:
                continue
            own_state = self.robots[i].state
            own_error = own_state - self.desired_states[i]

            own_pos = own_state[[0, 2]]


            # -------------------------
            # sensing initialization
            # -------------------------

            rel_obs = np.array([0.0, 0.0])
            dist_obs = self.sensing_radius

            rel_robot = np.array([0.0, 0.0])
            dist_robot = self.robot_sensing_radius

            # -------------------------
            # nearest robot sensing
            # -------------------------

            for j in range(self.num_of_bots):

                if j == i:
                    continue

                other_pos = self.robots[j].state[[0, 2]]
                d = np.linalg.norm(other_pos - own_pos)

                if d < dist_robot:
                    dist_robot = d
                    rel_robot = other_pos - own_pos
            
            # -------------------------
            # nearest obstacle sensing
            # -------------------------

            if len(self.obstacles) > 0:

                obstacle_array = np.asarray(
                    self.obstacles,
                    dtype=float
                )

                distances = np.linalg.norm(
                    obstacle_array - own_pos,
                    axis=1
                )

                nearest_idx = np.argmin(distances)
                nearest_dist = distances[nearest_idx]

                if nearest_dist <= self.sensing_radius:

                    nearest_obstacle = obstacle_array[nearest_idx]

                    rel_obs = nearest_obstacle - own_pos
                    dist_obs = nearest_dist

            self.nearest_robot_dist[i] = dist_robot
            self.nearest_obstacle_dist[i] = dist_obs

            x_normalize = 100.0
            y_normalize = 50.0

            obs[i] = np.array([
            own_error[0] / x_normalize,  # x position error
            own_error[1] / y_normalize,   # x velocity error
            own_error[2] / x_normalize,  # y position error
            own_error[3] / y_normalize,    # y velocity error

            rel_obs[0] / self.sensing_radius,
            rel_obs[1] / self.sensing_radius,
            dist_obs / self.sensing_radius,

            rel_robot[0] / self.robot_sensing_radius,
            rel_robot[1] / self.robot_sensing_radius,
            dist_robot / self.robot_sensing_radius
            ], dtype=np.float32)

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

    def spawn_fixed_obstacle(self):

        travel_dir = (
            self.FORMATION_VELOCITY.astype(float)
            / np.linalg.norm(self.FORMATION_VELOCITY)
        )

        obstacle_center = (
            self.formation_anchor
            + self.fixed_obstacle_offset * travel_dir
        )

        self.obstacles = [
            obstacle_center
        ]

        self.obstacles_radius = self.fixed_obstacle_radius

        self.obstacle_spawn_count += 1
        self.last_obstacle_spawn_step = self.current_step

