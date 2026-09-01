import random
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import env_20bots_no_tran as env
import os
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
# Simulation Parameters
from transformer_extractor import RobotTransformerExtractor

hist = {"t": [], "x": [], "y": [], "anchor_x": [], "anchor_y": []}
# Create a DataFrame to store the history of robot positions
WIDTH = 1980
HEIGHT = 1080
myenv = None
class Robot:
    def __init__(self, idx, pos):
        self.idx = idx
        # state: [x, dx, y, dy]
        self.state = np.array([pos[0], 0, pos[1], 0], dtype=float)
        # self.history = [pos]
        self.x_speed = self.state[1]
        self.y_speed = self.state[3]
        self.action = [0,0]
        self.neighbor_indexs = []

    def get_obs(self, neighbors, desired_states):
        q_all = np.array([r.state for r in neighbors])

        # Flatten q_all and desired_states for the full multi-robot state and desired state
        # q_all is now a 12x1 vector (3 robots, each with 4 states)
        error = q_all.flatten() - desired_states.flatten()
        return error
        
    # Draw the robot as a circle
    def draw(self, screen):
        x, y = int(self.state[0]), int(self.state[2])
        pygame.draw.circle(screen, (0, 255, 0), (x, y), 10)

def run_sim():
    pygame.init()
    total_steps = 50000
    angle = 0.0  # Initial angle for rotation
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    myenv = env.FormationEnv(screen)
    num_robots = myenv.num_of_bots    # leader_robot = robots[0]

    policy_kwargs = dict(
    features_extractor_class=RobotTransformerExtractor,
    features_extractor_kwargs=dict(
        features_dim=64,
        d_model=64,
        nhead=4,
        num_layers=2
    ),
    net_arch=[128, 128]
)


    # if you want to train from scratch, use the line below to create a new model. Otherwise, load a pre-trained model with the lines below that.
#     model = SAC(
#     "MlpPolicy",
#     myenv,
#     policy_kwargs=policy_kwargs,    # <-- THIS is where you use it
#     verbose=1,
#     tensorboard_log="./SAC_formation_env/"
# )

    model = SAC("MlpPolicy", myenv,verbose=1, tensorboard_log="./SAC_formation_env/")

    # Load a pre-trained model (make sure to adjust the path and filename as needed), only one of the lines below should be uncommented at a time, depending on which model you want to load

    #model = PPO.load("demo/no_target_ppo/test_4711_400k.zip",myenv, tensorboard_log="./sac_car_env/")
    #model = SAC.load("models/sac_gamma_5bots_tran_II/test_200k.zip",myenv, tensorboard_log="./sac_car_env/")
    #model = SAC.load("models/sac_gamma_7bots_tran_II/test_200k.zip",myenv, tensorboard_log="./sac_car_env/")
    #model = SAC.load("models/sac_gamma_10bots_tran_I/test_50k.zip",myenv, tensorboard_log="./sac_car_env/")
    #model = SAC.load("models/sac_gamma_10bots_tran_I/test_100k.zip",myenv, tensorboard_log="./sac_car_env/")
    #model = SAC.load("models/sac_gamma_15bots_tran_I/test_125k.zip",myenv, tensorboard_log="./sac_car_env/")
    #model = SAC.load("models/sac_gamma_20bots_tran_II/test_150k.zip",myenv, tensorboard_log="./sac_car_env/")
    model = SAC.load("models/sac_gamma_20bots_no_tran_I/test_125k.zip",myenv, tensorboard_log="./sac_car_env/")
    #model = SAC.load("models/sac_gamma_5bots_III/test_250k.zip",myenv, tensorboard_log="./sac_car_env/")
    #model = DDPG.load("demo/no_target_ddpg/test_4711_400k.zip",myenv, tensorboard_log="./DDPG_formation_env/")

    # train a model

    # model.learn(total_steps)
    # model.save(f"models/test_trace_4711/test_trace_4711_300k")

    # training in increments and saving intermediate models
    # for i in range(1,9):
    #     model.learn(total_timesteps=25000, reset_num_timesteps=False)
    #     model.save(f"models/sac_gamma_20bots_no_tran_I/test_{i*25}k")
    clock = pygame.time.Clock()

    # Main Loop
    running = True
    # reset env before start
    obs, info = myenv.reset()

    # print("Environment obs shape:", obs.shape)
    # print(model.policy)

    # print(obs.shape)
    # print(obs[:5])
    # print(obs[5:])

    while running:
        elaspsed_time = pygame.time.get_ticks() / 1000.0  # Get elapsed time in seconds
        hist["t"].append(elaspsed_time)
        hist["x"].append([r.state[0] for r in myenv.robots])
        hist["y"].append([r.state[2] for r in myenv.robots])
        hist["anchor_x"].append(myenv.formation_anchor[0])
        hist["anchor_y"].append(myenv.formation_anchor[1])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((30, 30, 30))
  
        pygame.draw.circle(screen, (255, 0, 0), (int(myenv.formation_anchor[0]), int(myenv.formation_anchor[1])), 10)
        

        #print(f"Desired states: {desired_states.shape}")


        THRESHOLD = 0.0  # tweak as needed
        # pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(myenv.target[0] - 5, myenv.target[1] - 5, 30, 30))

        info = pygame.display.Info()
        action, _states = model.predict(obs, deterministic = True)
        # print("action:", action)
        # print("action shape:", action.shape)
        obs, reward, terminated, end, info = myenv.step(action)
        
        myenv.render()
        total_steps += 1  # Increment step count
        if total_steps % 24000 == 0:
            # print("Training... at step ", total_steps)
            obs, info = myenv.reset()
        if terminated:
            obs, _ = myenv.reset()

    
            
    # return num_robots
    gamma_history = np.asarray(myenv.gamma_history, dtype=float)
    formation_error_history = np.asarray(
        myenv.formation_error_history,
        dtype=float
    )

    myenv.close()

    return num_robots, gamma_history, formation_error_history, myenv.FORMATION_OFFSET.copy()
        

if __name__ == "__main__":
    # Initialize Pygame and robots
    pygame.init()
    # creating robots with initial positions
    # num_robots = run_sim()
    num_robots, gamma_history, formation_error_history, formation_offset = run_sim()


    pygame.quit()
    hist["t"] = np.array(hist["t"])
    hist["x"] = np.array(hist["x"])
    hist["y"] = np.array(hist["y"])

    plt.figure(figsize=(8, 8))

    # robot trajectories
    for j in range(num_robots):
        plt.plot(hist["x"][:, j], hist["y"][:, j], label=f"Robot {j+1}")
        plt.scatter(hist["x"][0, j], hist["y"][0, j], marker="o")   # start
        plt.scatter(hist["x"][-1, j], hist["y"][-1, j], marker="x") # end

    # anchor trajectory
    plt.plot(hist["anchor_x"], hist["anchor_y"], "k--", linewidth=2, label="Anchor path")
    plt.scatter(hist["anchor_x"][0], hist["anchor_y"][0], marker="s", s=80, label="Anchor start")
    plt.scatter(hist["anchor_x"][-1], hist["anchor_y"][-1], marker="D", s=80, label="Anchor end")

    # final desired formation points
    final_anchor = np.array([hist["anchor_x"][-1], hist["anchor_y"][-1]])
    FORMATION_OFFSET = formation_offset

    for i, offset in enumerate(FORMATION_OFFSET):
        desired_point = final_anchor + offset
        plt.scatter(desired_point[0], desired_point[1], marker="*", s=150, label=f"Desired {i+1}")

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Robot Trajectories with Anchor and Desired Formation")
    plt.legend(loc="best")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("robot_trajectories_with_anchor.png", dpi=160)
    plt.show()

        # fig = plt.figure(figsize=(8, 5))
        # for j in range(num_robots):
        #     plt.plot(hist['t'], hist['x'][:, j], label=f"x{j+1}")
        # plt.xlabel("time [s]")
        # plt.ylabel("x position")
        # plt.title("x(t) with formation offsets")
        # plt.legend(loc="best")
        # fig.tight_layout()
        # plt.savefig("formation_x.png", dpi=160)
        # plt.close(fig)

        # fig2 = plt.figure(figsize=(8, 5))
        # for j in range(num_robots):
        #     plt.plot(hist['t'], hist['y'][:, j], label=f"y{j+1}")
        # plt.xlabel("time [s]")
        # plt.ylabel("y position")
        # plt.title("y(t) converging to formation offsets")
        # plt.legend(loc="best")
        # fig2.tight_layout()
        # plt.savefig("formation_y.png", dpi=160)
        # plt.close(fig2)

    # =========================================================
# Plot Gamma values
# =========================================================

if len(gamma_history) > 0:
    steps = np.arange(len(gamma_history))

    plt.figure(figsize=(10, 5))

    plt.plot(
        steps,
        gamma_history[:, 0],
        label="g1"
    )

    plt.plot(
        steps,
        gamma_history[:, 1],
        label="g2"
    )

    plt.xlabel("Simulation Step")
    plt.ylabel("Gamma Value")
    plt.title("Gamma Values Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "gamma_history.png",
        dpi=160
    )

    plt.show()


# =========================================================
# Plot formation error
# =========================================================

if len(formation_error_history) > 0:
    steps = np.arange(len(formation_error_history))

    plt.figure(figsize=(10, 5))

    plt.plot(
        steps,
        formation_error_history,
        label="Formation Error"
    )

    plt.xlabel("Simulation Step")
    plt.ylabel("Formation Error")
    plt.title("Formation Error Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "formation_error_history.png",
        dpi=160
    )

    plt.show()

    print("Plots saved: formation_x.png and formation_y.png")