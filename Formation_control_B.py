import random
import pygame
import numpy as np
import time
from scipy.linalg import eigvals
import matplotlib.pyplot as plt


# Simulation Parameters
WIDTH, HEIGHT = 1980, 1080
FPS = 60
NUM_ROBOTS = 3
DT = 0.01
FORMATION_VELOCITY = np.array([0.5, 0.5])

# try to make a triangle
FORMATION_OFFSET = np.array([[0, 0], [20, 100], [100,50]])
hist = {"t": [], "x": [], "y": []}

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
# L = Degree_matrix - A
print("Laplacian Matrix L:\n", L)

I_m = np.eye(3)

# Sytem dynamics matrices for a double integrator model
# A0 is the state transition(from current state to next state x->dx) matrix for a single robot
# B0 is the input matrix for a single robot
# Each robot has a state of [x, dx, y, dy] (position and velocity in 2D)
A0 = np.array([
    [0, 1, 0, 0], 
    [0, 0, 0, 0], 
    [0, 0, 0, 1], 
    [0, 0, 0, 0]  
])

B0 = np.array([
    [0, 0],
    [1, 0],  # dx affected by u_x
    [0, 0],
    [0, 1]   # dy affected by u_y
])


# For u2_i = [control_y, control_dy]
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


# L1 = L

# we only have 2 g since our sytem is 2D
# Gamma_1 = [
#     [-g₁, -g₂,  0,    0   ],
#     [ 0,    0,  -g₁, -g₂ ]
# ]

Gamma_1 = [
    [-0.9090909090909136, -1.5151515151515156,  0.0,  0.0], #gain control feedback for x-driection
    [  0.0,  0.0, -0.9090909090909136, -1.5151515151515156] #gain control feedback for y-driection
]


Gamma = np.array(Gamma_1) * 0.1  # shape (2×4), controls x and y
alpha = np.exp(1)
u1s = []

y1 = []

def alpha_derivative(t, alpha0=0.05, decay=0.1):
    """
    Positive, decaying time-scaling derivative
    """
    # return alpha0 * np.exp(-decay * t)
    return alpha0 * t


def compute_y1(robots, desired_states):
    return np.array([r.state[0] - desired_states[r.idx][0] - alpha for r in robots])



class Robot:
    def __init__(self, idx, pos):
        self.idx = idx
        # state: [x, dx, y, dy]
        self.state = np.array([pos[0], 0, pos[1], 0], dtype=float)

    def update(self, u2):
        # u2 shape: (3 robots x 2 control inputs)
        u2_i = u2[self.idx]           # (2,) for this robot
        # print(f"Robot {self.idx} control input: {u2_i}")

        # Reshape to column vector (2x1)
        u_input = u2_i.reshape(2,1)
        u_input += FORMATION_VELOCITY.reshape(2,1)

        # Full 4D state
        state = self.state.reshape(4,1)

        # Apply dynamics
        dq = A0 @ state + B0 @ u_input  # (4x1)

        # Update state
        self.state += (dq * DT).flatten()


    def draw(self, screen, offset_x=0, offset_y=0):
        SCALE = 0.1
        x = float(self.state[0]) * SCALE
        y = float(self.state[2]) * SCALE

        # draw_x = int(x - offset_x * SCALE) + WIDTH // 2
        # draw_y = int(y - offset_y * SCALE) + HEIGHT // 2
        
        # pygame.draw.circle(screen, (0, 255, 0), (draw_x, draw_y), 10)
        x, y = int(self.state[0]), int(self.state[2])
        pygame.draw.circle(screen, (0, 255, 0), (x, y), 10)

# Initialize Pygame and robots
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
start_time = time.time()

# creating robots with initial positions
robots = [
    Robot(i, (
        np.random.uniform(10, WIDTH -100),   # random x within (10, 100)
        np.random.uniform(10, HEIGHT - 100)   # random y within bounds
    ))
    for i in range(NUM_ROBOTS)
]
# robots = [
#     Robot(i, (
#        0,   # random x within (10, 100)
#         0   # random y within bounds
#     ))
#     for i in range(NUM_ROBOTS)
# ]

angle = 0.0  # Initial angle for rotation
TARGET = np.array([random.uniform(100, WIDTH - 100), random.uniform(100, HEIGHT - 100)]) 
leader_robot = robots[0]
formation_anchor = robots[0].state[[0, 2]]  # Anchor position is the first robot's position
desired_states = np.array([
[offset[0], 0, offset[1], 0]
    for offset in FORMATION_OFFSET
])

# Check A1 + λ B1 Gamma1 for each nonzero eigenvalue
eig_LG = np.linalg.eigvals(L)
nonzero_lams = [lam for lam in eig_LG if abs(lam) > 1e-8]
for lam in nonzero_lams:
    if lam != 0:
        M = A0 + lam * B0 @ Gamma_1
        eig_M = eigvals(M)
        print(f"lambda = {lam:.3f}, eigenvalues of A1 + λ B1Γ1 = {eig_M}")
        if np.all(np.real(eig_M) < 0):
            print("Hurwitz ✅")
        else:
            print("Not Hurwitz ❌")

# Main Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    center_x = np.mean([r.state[0] for r in robots])
    center_y = np.mean([r.state[2] for r in robots])

    # Compute offset to keep center in middle of screen
    offset_x = center_x - WIDTH // 2
    offset_y = center_y - HEIGHT // 2
    screen.fill((30, 30, 30))
    elaspsed_time = pygame.time.get_ticks() / 1000.0  # Get elapsed time in seconds
    hist["t"].append(elaspsed_time)
    hist["x"].append([r.state[0] for r in robots])  # all x
    hist["y"].append([r.state[2] for r in robots])  # all y

    
    velocity = np.array([0.5, 0.5])
    FORMATION_VELOCITY = velocity

    formation_anchor += FORMATION_VELOCITY * DT

    desired_states = np.array([
    [formation_anchor[0] + offset[0], 0, formation_anchor[1] + offset[1], 0]
        for offset in FORMATION_OFFSET
    ])

    THRESHOLD = 0.0  # tweak as needed
    # pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(TARGET[0] - 5, TARGET[1] - 5, 30, 30))

    y1 = compute_y1(robots, desired_states)

    y1 = y1.reshape(-1, 1)
    # x difference for all robots
    u1 = alpha_derivative(elaspsed_time) * np.ones((NUM_ROBOTS, 1)) - L @ y1
    u1 = np.maximum(u1, 0.01)

    u1_diag = np.diag(u1.flatten())  # shape (3, 3) for 3 robots

    z_full = np.array([r.state for r in robots])          # shape (3,4)
    # h_full: (3,4) with [x_d, 0, y_d, 0]
    # but the paper requires h = [0, 0, y_d, 0]
    h_full = np.zeros_like(z_full)
    h_full[:, 0] = desired_states[:, 0]  # x_d
    h_full[:, 2] = desired_states[:, 2]  # y_d

    error_full = z_full - h_full         # (3,4)
    rho = L1 @ error_full.flatten()      # (12,)

    u1_gamma = np.kron(u1_diag, Gamma) 

    u2 = u1_gamma @ rho.flatten()
    u2 = u2.reshape(3, 2)  # (6,)


    for i, robot in enumerate(robots):
        robot.update(u2)
        robot.draw(screen, offset_x, offset_y)

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
plt.savefig("formation_x_approach_2.png", dpi=160)
plt.close(fig)

fig2 = plt.figure(figsize=(8, 5))
for j in range(NUM_ROBOTS):
    plt.plot(hist['t'], hist['y'][:, j], label=f"y{j+1}")
plt.xlabel("time [s]")
plt.ylabel("y position")
plt.title("y(t) converging to formation offsets")
plt.legend(loc="best")
fig2.tight_layout()
plt.savefig("formation_y_approach_2.png", dpi=160)
plt.close(fig2)

print("Plots saved: formation_x_approach_2.png and formation_y_approach_2.png")