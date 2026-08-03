import random
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import env
import os
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
# Simulation Parameters

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

    # def update(self, neighbors, desired_states):
        # Build q (stack of all robot states, each 1x4), will get 3x4 matrix
        # q_all = np.array([r.state for r in neighbors])

        # Flatten q_all and desired_states for the full multi-robot state and desired state
        # q_all is now a 12x1 vector (3 robots, each with 4 states)
        # error = q_all.flatten() - desired_states.flatten()

        # Apply Laplacian to error
        # rho = L1 @ error

        # # the control law for all robots
        # r = Gamma @ rho

        # Feedforward + anchor tracking
        # r_i = r[self.idx * 2 : self.idx * 2 + 2]
        # r_i += FORMATION_VELOCITY  # Add anchor's desired velocity

        # LEADER_IDX = 1 #optional leader role
        # if (self.idx == LEADER_IDX):  # If this is the second robot
        #     # Add global error correction toward desired position
        #     Kp_global = 5.0
        #     pos = self.state[[0, 2]]  # current (x, y)
        #     desired_pos = desired_states[self.idx][[0, 2]]
        # #print(f"before adding global error correction: r_i={r_i}, desired_pos={desired_pos}, pos={pos}")
        #     r_i += Kp_global * (desired_pos - pos)
        # #print(f"after adding global error correction: r_i={r_i}")

        # # Kd_damping = 5  # Tune this value to reduce overshoot
        # # velocity = self.state[[1, 3]]  # (vx, vy)
        # # r_i -= Kd_damping * velocity

        # r_i = self.action
        # r_i += FORMATION_VELOCITY
        # dq = A0 @ self.state.reshape(4, 1) + B0 @ r_i.reshape(2, 1)
        # self.state += dq.flatten() * DT
        #print(f"Robot {self.idx} -  ri: {dq.flatten() * DT}")

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

def save_episode(myenv, seed, controller_name):
    gamma = np.array(myenv.gamma_history)
    error = np.array(myenv.formation_error_history)

    df = pd.DataFrame({
        "step": np.arange(len(error)),
        "g1": gamma[:, 0],
        "g2": gamma[:, 1],
        "formation_error": error
    })

    filename = f"evaluation_logs/{controller_name}_seed_{seed}.csv"
    df.to_csv(filename, index=False)

    print(f"Saved {filename}")

def run_sim():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    #myenv = env.FormationEnv(screen)
    myenv = env.FormationEnv(None, render_mode=False)    
    num_robots = myenv.num_of_bots

    controller_name = "best_controller"

    os.makedirs("evaluation_logs", exist_ok=True)

    # model = SAC.load(
    #     "models/sac_gamma_9403001_VIIII/test_550k.zip",
    #     myenv,
    #     tensorboard_log="./sac_car_env/"
    # )

    model = SAC.load("models/sac_gamma_9403001_V/test_400k.zip",myenv, tensorboard_log="./sac_car_env/"
)

    seeds = range(30)  # use range(30) after debugging

    all_gamma_history = []
    all_formation_error_history = []

    for seed in seeds:
        print(f"Evaluating seed {seed}")

        obs, info = myenv.reset(seed=seed)

        terminated = False
        truncated = False

        while not terminated and not truncated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    break

            screen.fill((30, 30, 30))

            pygame.draw.circle(
                screen,
                (255, 0, 0),
                (
                    int(myenv.formation_anchor[0]),
                    int(myenv.formation_anchor[1])
                ),
                10
            )

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = myenv.step(action)

            myenv.render()

        # Save only after this episode has ended
        save_episode(
            myenv=myenv,
            seed=seed,
            controller_name=controller_name
        )

        gamma = np.asarray(myenv.gamma_history, dtype=float)
        error = np.asarray(
            myenv.formation_error_history,
            dtype=float
        )

        all_gamma_history.append(gamma)
        all_formation_error_history.append(error)

    myenv.close()

    return (
        num_robots,
        all_gamma_history,
        all_formation_error_history
    )
    

if __name__ == "__main__":
    # Initialize Pygame and robots
    pygame.init()
    # creating robots with initial positions
    num_robots, gamma_history, formation_error_history = run_sim()


    pygame.quit()
    # hist["t"] = np.array(hist["t"])
    # hist["x"] = np.array(hist["x"])
    # hist["y"] = np.array(hist["y"])

    # plt.figure(figsize=(8, 8))

    # robot trajectories
    # for j in range(num_robots):
    #     plt.plot(hist["x"][:, j], hist["y"][:, j], label=f"Robot {j+1}")
    #     plt.scatter(hist["x"][0, j], hist["y"][0, j], marker="o")   # start
    #     plt.scatter(hist["x"][-1, j], hist["y"][-1, j], marker="x") # end

    # # anchor trajectory
    # plt.plot(hist["anchor_x"], hist["anchor_y"], "k--", linewidth=2, label="Anchor path")
    # plt.scatter(hist["anchor_x"][0], hist["anchor_y"][0], marker="s", s=80, label="Anchor start")
    # plt.scatter(hist["anchor_x"][-1], hist["anchor_y"][-1], marker="D", s=80, label="Anchor end")

    # # final desired formation points
    # final_anchor = np.array([hist["anchor_x"][-1], hist["anchor_y"][-1]])
    # FORMATION_OFFSET = np.array([[0, 0], [20, 100], [100,50]])
    # for i, offset in enumerate(FORMATION_OFFSET):
    #     desired_point = final_anchor + offset
    #     plt.scatter(desired_point[0], desired_point[1], marker="*", s=150, label=f"Desired {i+1}")

    # plt.xlabel("X position")
    # plt.ylabel("Y position")
    # plt.title("Robot Trajectories with Anchor and Desired Formation")
    # plt.legend(loc="best")
    # plt.grid(True)
    # plt.axis("equal")
    # plt.tight_layout()
    # plt.savefig("robot_trajectories_with_anchor.png", dpi=160)
    # plt.show()


    # =========================================================
    # Plot Gamma values
    # =========================================================

    # if len(gamma_history) > 0:
    #     steps = np.arange(len(gamma_history))

    #     plt.figure(figsize=(10, 5))

    #     plt.plot(
    #         steps,
    #         gamma_history[:, 0],
    #         label="g1"
    #     )

    #     plt.plot(
    #         steps,
    #         gamma_history[:, 1],
    #         label="g2"
    #     )

    #     plt.xlabel("Simulation Step")
    #     plt.ylabel("Gamma Value")
    #     plt.title("Gamma Values Over Time")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()

    #     plt.savefig(
    #         "gamma_history.png",
    #         dpi=160
    #     )

    #     plt.show()


    # # =========================================================
    # # Plot formation error
    # # =========================================================

    # if len(formation_error_history) > 0:
    #     steps = np.arange(len(formation_error_history))

    #     plt.figure(figsize=(10, 5))

    #     plt.plot(
    #         steps,
    #         formation_error_history,
    #         label="Formation Error"
    #     )

    #     plt.xlabel("Simulation Step")
    #     plt.ylabel("Formation Error")
    #     plt.title("Formation Error Over Time")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()

    #     plt.savefig(
    #         "formation_error_history.png",
    #         dpi=160
    #     )

    #     plt.show()
    #     print("Plots saved: formation_x.png and formation_y.png")
