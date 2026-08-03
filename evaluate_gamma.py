import argparse
import contextlib
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pygame
from stable_baselines3 import SAC

import env


DEFAULT_MODEL = "models/sac_gamma_9403001_VIIII/test_550k.zip"


def classify_result(result):
    formation = result["final_formation_error"]
    tracking = result["final_tracking_error"]

    if result["truncated"]:
        return "failure"
    if formation < 0.8 and tracking < 0.8:
        return "high_quality_success"
    if formation < 1.2 and tracking < 1.2:
        return "acceptable_success"
    if formation < 1.5 or tracking < 1.5:
        return "partial_success"
    return "failure"


def run_episode(model, episode_idx, seed, max_steps, quiet_env=True):
    np.random.seed(seed)
    random.seed(seed)

    with contextlib.ExitStack() as stack:
        if quiet_env:
            stream = stack.enter_context(open(os.devnull, "w"))
            stack.enter_context(contextlib.redirect_stdout(stream))

        screen = pygame.Surface((1980, 1080))
        sim_env = env.FormationEnv(screen)
        obs, _ = sim_env.reset(seed=seed)

        history = {
            "x": [[robot.state[0] for robot in sim_env.robots]],
            "y": [[robot.state[2] for robot in sim_env.robots]],
            "anchor_x": [float(sim_env.formation_anchor[0])],
            "anchor_y": [float(sim_env.formation_anchor[1])],
            "formation_error": [],
            "tracking_error": [],
        }

        truncated = False
        terminated = False
        steps = 0

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = sim_env.step(action)
            steps = step + 1

            history["x"].append([robot.state[0] for robot in sim_env.robots])
            history["y"].append([robot.state[2] for robot in sim_env.robots])
            history["anchor_x"].append(float(sim_env.formation_anchor[0]))
            history["anchor_y"].append(float(sim_env.formation_anchor[1]))
            history["formation_error"].append(float(sim_env.formation_error))
            history["tracking_error"].append(float(sim_env.tracking_error))

            if terminated or truncated:
                break

    final_formation_error = history["formation_error"][-1]
    final_tracking_error = history["tracking_error"][-1]

    result = {
        "episode": episode_idx,
        "seed": seed,
        "steps": steps,
        "final_formation_error": final_formation_error,
        "final_tracking_error": final_tracking_error,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "history": history,
    }
    return result


def save_metrics_csv(results, output_dir):
    csv_path = output_dir / "episode_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "seed",
                "steps",
                "final_formation_error",
                "final_tracking_error",
                "outcome",
                "terminated",
                "truncated",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({k: result[k] for k in writer.fieldnames})
    return csv_path


def save_summary_json(results, args, output_dir):
    formation = np.array([r["final_formation_error"] for r in results], dtype=float)
    tracking = np.array([r["final_tracking_error"] for r in results], dtype=float)
    outcome_counts = {
        outcome: sum(r["outcome"] == outcome for r in results)
        for outcome in [
            "high_quality_success",
            "acceptable_success",
            "partial_success",
            "failure",
        ]
    }
    outcome_rates = {
        outcome: count / len(results)
        for outcome, count in outcome_counts.items()
    }

    summary = {
        "model": args.model,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "outcome_rules": {
            "high_quality_success": "formation < 0.8 and tracking < 0.8",
            "acceptable_success": "formation < 1.2 and tracking < 1.2",
            "partial_success": "formation < 1.5 or tracking < 1.5",
            "failure": "truncated/diverged, or none of the above",
        },
        "outcome_counts": outcome_counts,
        "outcome_rates": outcome_rates,
        "final_formation_error": {
            "mean": float(formation.mean()),
            "std": float(formation.std(ddof=0)),
            "min": float(formation.min()),
            "max": float(formation.max()),
        },
        "final_tracking_error": {
            "mean": float(tracking.mean()),
            "std": float(tracking.std(ddof=0)),
            "min": float(tracking.min()),
            "max": float(tracking.max()),
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path, summary


def save_plots(results, output_dir):
    traj_path = output_dir / "trajectories.png"
    error_path = output_dir / "errors_over_time.png"
    hist_path = output_dir / "final_error_histograms.png"

    plt.figure(figsize=(9, 8))
    for result in results:
        x = np.array(result["history"]["x"])
        y = np.array(result["history"]["y"])
        anchor_x = np.array(result["history"]["anchor_x"])
        anchor_y = np.array(result["history"]["anchor_y"])
        alpha = 0.25 if result["outcome"] == "failure" else 0.55
        for robot_idx in range(x.shape[1]):
            plt.plot(x[:, robot_idx], y[:, robot_idx], alpha=alpha)
            plt.scatter(x[0, robot_idx], y[0, robot_idx], s=12, marker="o", alpha=alpha)
            plt.scatter(x[-1, robot_idx], y[-1, robot_idx], s=18, marker="x", alpha=alpha)
        plt.plot(anchor_x, anchor_y, "k--", linewidth=1, alpha=0.18)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Gamma SAC Evaluation: Robot Trajectories")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(traj_path, dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    for result in results:
        formation_error = np.array(result["history"]["formation_error"])
        tracking_error = np.array(result["history"]["tracking_error"])
        alpha = 0.25 if result["outcome"] == "failure" else 0.55
        plt.plot(formation_error, color="tab:blue", alpha=alpha)
        plt.plot(tracking_error, color="tab:orange", alpha=alpha)
    plt.xlabel("Step")
    plt.ylabel("Error")
    plt.title("Formation Error (blue) and Tracking Error (orange)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(error_path, dpi=180)
    plt.close()

    formation = [r["final_formation_error"] for r in results]
    tracking = [r["final_tracking_error"] for r in results]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(formation, bins=min(10, len(results)))
    plt.xlabel("Final formation error")
    plt.ylabel("Count")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.hist(tracking, bins=min(10, len(results)))
    plt.xlabel("Final tracking error")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(hist_path, dpi=180)
    plt.close()

    return traj_path, error_path, hist_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the current 3-robot Gamma SAC model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to SAC .zip checkpoint.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of random starts.")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum rollout steps per start.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--output-dir", default="eval_gamma_results")
    parser.add_argument("--verbose-env", action="store_true", help="Show env reward prints.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pygame.init()
    model = SAC.load(args.model)

    results = []
    for episode_idx in range(args.episodes):
        seed = args.seed + episode_idx
        result = run_episode(
            model=model,
            episode_idx=episode_idx,
            seed=seed,
            max_steps=args.max_steps,
            quiet_env=not args.verbose_env,
        )
        result["outcome"] = classify_result(result)
        results.append(result)
        print(
            f"episode {episode_idx:02d} seed={seed} steps={result['steps']} "
            f"formation={result['final_formation_error']:.4f} "
            f"tracking={result['final_tracking_error']:.4f} "
            f"outcome={result['outcome']}"
        )

    csv_path = save_metrics_csv(results, output_dir)
    summary_path, summary = save_summary_json(results, args, output_dir)
    plot_paths = save_plots(results, output_dir)

    print("\nSummary")
    for outcome, rate in summary["outcome_rates"].items():
        count = summary["outcome_counts"][outcome]
        print(f"{outcome}: {count}/{args.episodes} ({rate:.1%})")
    print(
        "final formation error: "
        f"mean={summary['final_formation_error']['mean']:.4f}, "
        f"std={summary['final_formation_error']['std']:.4f}"
    )
    print(
        "final tracking error: "
        f"mean={summary['final_tracking_error']['mean']:.4f}, "
        f"std={summary['final_tracking_error']['std']:.4f}"
    )
    print(f"saved metrics: {csv_path}")
    print(f"saved summary: {summary_path}")
    print("saved plots:")
    for path in plot_paths:
        print(f"  {path}")
    pygame.quit()


if __name__ == "__main__":
    main()
