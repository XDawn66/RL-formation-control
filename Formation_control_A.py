import random
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Simulation Parameters
WIDTH, HEIGHT = 1980, 1080
FPS = 60
NUM_ROBOTS = 3
DT = 0.03
FORMATION_VELOCITY = np.array([0.5, 0.5])

# try to make a triangle
FORMATION_OFFSET = np.array([[0, 0], [20, 100], [100,50]])
hist = {"t": [], "x": [], "y": []}
# Create a DataFrame to store the history of robot positions

# adjacency matrix for a directed graph
A = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

Degree_matrix = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])
 

#identity matrix
I = np.eye(3)

# Laplacian matrix 3x3
L = I - Degree_matrix @ A
print("Laplacian Matrix L:\n", L)

I_m = np.eye(3)

# Sytem dynamics matrices for a double integrator model
# A0 is the state transition(from current state to next state x->dx) matrix for a single robot
# B0 is the input matrix for a single robot
# Each robot has a state of [x, dx, y, dy] (position and velocity in 2D)
A0 = np.array([
    [0, 1, 0, 0],  # dx = vx
    [0, 0, 0, 0],  # dvx = ux (handled by B0)
    [0, 0, 0, 1],  # dy = vy
    [0, 0, 0, 0]   # dvy = uy (handled by B0)
])

B0 = np.array([
    [0, 0],  # input doesn't affect position directly
    [1, 0],  # u_x affects dvx/dt
    [0, 0],  # input doesn't affect position directly
    [0, 1]   # u_y affects dvy/dt
])
 #simple double integrator where only consider postion and velocity
 #we want to make sure robot's dynmics are indenpendent from each other,
 # since each robot has 4 states, we need to create a block diagonal matrix (3x3) x(4x4) for multiple robots
#Av is the whole picture of the system dynamics
#Bv is the input matrix for the whole system
A_v = np.kron(np.eye(NUM_ROBOTS), A0) # This will create a 12x12 matrix 

# For multiple robots (3 robots in this case), use the correct identity matrix for the number of states
B_v = np.kron(np.eye(NUM_ROBOTS), B0)  # This will create a 12x6 matrix

I_2m = np.eye(2 * 2)

# Turning the Laplacian matrix into a 12x12 matrix to allow further calculations
L1 = np.kron(L, np.eye(4))

# we only have 2 g since our sytem is 2D
# Gamma_1 = [
#     [-g₁, -g₂,  0,    0   ],
#     [ 0,    0,  -g₁, -g₂ ]
# ]
Gamma_1 = [
    [-50.0, -30.0,  0.0,  0.0], #gain control feedback for x-driection
    [  0.0,  0.0, -50.0, -30.0] #gain control feedback for y-driection
]
  # shape (2×4), controls x and y

Gamma = np.kron(np.eye(NUM_ROBOTS), Gamma_1)  # shape (6×12)

eigvals = np.linalg.eigvals(L)
lambda_ = 1.5
A_cl = A0 + lambda_ * B0 @ Gamma_1
np.linalg.eigvals(A_cl)
print("Eigenvalues of A_cl:", np.linalg.eigvals(A_cl))


class Robot:
    def __init__(self, idx, pos):
        self.idx = idx
        # state: [x, dx, y, dy]
        self.state = np.array([pos[0], 0, pos[1], 0], dtype=float)
        # self.history = [pos]
        self.x_speed = self.state[1]
        self.y_speed = self.state[3]

    def update(self, neighbors, desired_states):
        # Build q (stack of all robot states, each 1x4), will get 3x4 matrix
        q_all = np.array([r.state for r in neighbors])

        # Flatten q_all and desired_states for the full multi-robot state and desired state
        # q_all is now a 12x1 vector (3 robots, each with 4 states)
        error = q_all.flatten() - desired_states.flatten()

        # Apply Laplacian to error
        # rho = L1 @ error

        # # the control law for all robots
        # r = Gamma @ rho

        # Feedforward + anchor tracking
        # r_i = r[self.idx * 2 : self.idx * 2 + 2]
        # r_i += FORMATION_VELOCITY  # Add anchor's desired velocity

        LEADER_IDX = 1 #optional leader role
        if (self.idx == LEADER_IDX):  # If this is the second robot
            # Add global error correction toward desired position
            Kp_global = 5.0
            pos = self.state[[0, 2]]  # current (x, y)
            desired_pos = desired_states[self.idx][[0, 2]]
        #print(f"before adding global error correction: r_i={r_i}, desired_pos={desired_pos}, pos={pos}")
            r_i += Kp_global * (desired_pos - pos)
        #print(f"after adding global error correction: r_i={r_i}")

        # Kd_damping = 5  # Tune this value to reduce overshoot
        # velocity = self.state[[1, 3]]  # (vx, vy)
        # r_i -= Kd_damping * velocity

        r_i = action[self.idx * 2 : self.idx * 2 + 2]
        dq = A0 @ self.state.reshape(4, 1) + B0 @ r_i.reshape(2, 1)
        self.state += dq.flatten() * DT
        #print(f"Robot {self.idx} -  ri: {dq.flatten() * DT}")

    def get_obs(self, neighbors, desired_states):
        q_all = np.array([r.state for r in neighbors])

        # Flatten q_all and desired_states for the full multi-robot state and desired state
        # q_all is now a 12x1 vector (3 robots, each with 4 states)
        error = q_all.flatten() - desired_states.flatten()
        return error

    def get_reward(self):
        w1 = 1
        w2 = 1
        w3 = 1
        
        error = self.get_obs()
        formation_error = np.linalg.norm(L1 @ error)
        tracking_error = np.linalg.norm(error)
        control_effort = np.linalg.norm(action)
        reward = -w1 * formation_error**2 - w2 * tracking_error**2 - w3 * control_effort**2
        return reward
    
    # Draw the robot as a circle
    def draw(self, screen):
        x, y = int(self.state[0]), int(self.state[2])
        pygame.draw.circle(screen, (0, 255, 0), (x, y), 10)

# Initialize Pygame and robots
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# creating robots with initial positions
robots = [
    Robot(i, (
        np.random.uniform(10, WIDTH -100),   # random x within (10, 100)
        np.random.uniform(10, HEIGHT - 100)   # random y within bounds
    ))
    for i in range(NUM_ROBOTS)
]


angle = 0.0  # Initial angle for rotation
TARGET = np.array([random.uniform(100, WIDTH - 100), random.uniform(100, HEIGHT - 100)]) 
leader_robot = robots[0]
formation_anchor = robots[0].state[[0, 2]]  # Anchor position is the first robot's position

# Main Loop
running = True
while running:
    elaspsed_time = pygame.time.get_ticks() / 1000.0  # Get elapsed time in seconds
    hist["t"].append(elaspsed_time)
    hist["x"].append([r.state[0] for r in robots])  # all x
    hist["y"].append([r.state[2] for r in robots])  # all y
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((30, 30, 30))

    direction = TARGET - formation_anchor
    distance = np.linalg.norm(direction)

    if distance > 5:  # only move if not yet at target
        velocity = direction / distance * 40  # control speed (tune 30)
    else:
        velocity = np.array([0.0, 0.0])  # stop when close enough

    FORMATION_VELOCITY = velocity

    formation_anchor += FORMATION_VELOCITY * DT
    pygame.draw.circle(screen, (255, 0, 0), (int(formation_anchor[0]), int(formation_anchor[1])), 10)
    
    # 3x4 matrix for desired states

    desired_states = np.array([
    [formation_anchor[0] + offset[0], 0, formation_anchor[1] + offset[1], 0]
        for offset in FORMATION_OFFSET
    ])
    #print(f"Desired states: {desired_states.shape}")


    THRESHOLD = 0.0  # tweak as needed
    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(TARGET[0] - 5, TARGET[1] - 5, 30, 30))

    for i, robot in enumerate(robots):
        robot.update(robots, desired_states)
        robot.draw(screen)


    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
hist["t"] = np.array(hist["t"])
hist["x"] = np.array(hist["x"])
hist["y"] = np.array(hist["y"])
fig = plt.figure(figsize=(8, 5))
for j in range(NUM_ROBOTS):
    plt.plot(hist['t'], hist['x'][:, j], label=f"x{j+1}")
plt.xlabel("time [s]")
plt.ylabel("x position")
plt.title("x(t) with formation offsets")
plt.legend(loc="best")
fig.tight_layout()
plt.savefig("formation_x.png", dpi=160)
plt.close(fig)

fig2 = plt.figure(figsize=(8, 5))
for j in range(NUM_ROBOTS):
    plt.plot(hist['t'], hist['y'][:, j], label=f"y{j+1}")
plt.xlabel("time [s]")
plt.ylabel("y position")
plt.title("y(t) converging to formation offsets")
plt.legend(loc="best")
fig2.tight_layout()
plt.savefig("formation_y.png", dpi=160)
plt.close(fig2)

print("Plots saved: formation_x.png and formation_y.png")