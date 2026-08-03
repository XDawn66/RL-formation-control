from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOG_DIR = Path("evaluation_logs")
OUTPUT_DIR = Path("analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

CONTROLLERS = [
    "best_controller",
    "second_best_controller",
]

# dt = 0.005:
# 50 steps = 0.25 s
# 100 steps = 0.50 s
# 200 steps = 1.00 s
FUTURE_WINDOWS = [50, 100, 200]


def load_controller_data(controller_name: str) -> list[pd.DataFrame]:
    files = sorted(LOG_DIR.glob(f"{controller_name}_seed_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found for {controller_name} in {LOG_DIR}"
        )

    episodes = []

    for file in files:
        df = pd.read_csv(file)

        required_columns = {
            "step",
            "g1",
            "g2",
            "formation_error",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{file} is missing columns: {sorted(missing)}"
            )

        # Derive the seed from the filename
        seed_text = file.stem.split("_seed_")[-1]
        df["seed"] = int(seed_text)
        df["controller"] = controller_name

        episodes.append(df)

    return episodes


def analyze_episode(
    episode: pd.DataFrame,
    future_window: int,
) -> pd.DataFrame:
    g1 = episode["g1"].to_numpy(dtype=float)
    g2 = episode["g2"].to_numpy(dtype=float)
    error = episode["formation_error"].to_numpy(dtype=float)

    # Need t-1 and t+future_window to both exist
    if len(episode) <= future_window + 1:
        return pd.DataFrame()

    gamma = np.column_stack([g1, g2])

    # delta_gamma[t] = ||Gamma[t] - Gamma[t-1]||
    delta_gamma = np.linalg.norm(
        gamma[1:] - gamma[:-1],
        axis=1,
    )

    # For time t = 1 through len(error)-future_window-1
    current_error = error[1:-future_window]

    future_error = error[
        1 + future_window:
    ]

    future_error_change = future_error - current_error

    # Match delta_gamma to those same t indices
    delta_gamma = delta_gamma[:-future_window]

    if not (
        len(delta_gamma)
        == len(current_error)
        == len(future_error_change)
    ):
        raise RuntimeError("Analysis arrays are not aligned")

    return pd.DataFrame({
        "seed": int(episode["seed"].iloc[0]),
        "controller": episode["controller"].iloc[0],
        "step": episode["step"].to_numpy()[1:-future_window],
        "gamma_change": delta_gamma,
        "current_error": current_error,
        "future_error": future_error,
        "future_error_change": future_error_change,
        "improved": future_error_change < 0,
        "future_window": future_window,
    })


def summarize_controller(
    controller_name: str,
    data: pd.DataFrame,
) -> dict:
    if data.empty:
        raise ValueError(f"No analysis data for {controller_name}")

    # Define a "large" Gamma change relative to this controller
    large_threshold = data["gamma_change"].quantile(0.75)
    large = data[data["gamma_change"] >= large_threshold]
    small = data[data["gamma_change"] < large_threshold]

    spearman = data["gamma_change"].corr(
        data["future_error_change"],
        method="spearman",
    )

    return {
        "controller": controller_name,
        "future_window": int(data["future_window"].iloc[0]),
        "samples": len(data),
        "large_change_threshold": large_threshold,
        "spearman_gamma_vs_future_error": spearman,
        "mean_future_change_all": data["future_error_change"].mean(),
        "mean_future_change_large_gamma": (
            large["future_error_change"].mean()
        ),
        "mean_future_change_small_gamma": (
            small["future_error_change"].mean()
        ),
        "improvement_rate_all": data["improved"].mean(),
        "improvement_rate_large_gamma": large["improved"].mean(),
        "improvement_rate_small_gamma": small["improved"].mean(),
    }


all_analyses = []
summary_rows = []

for controller in CONTROLLERS:
    episodes = load_controller_data(controller)

    print(f"{controller}: loaded {len(episodes)} episodes")

    for window in FUTURE_WINDOWS:
        episode_results = [
            analyze_episode(episode, window)
            for episode in episodes
        ]

        controller_data = pd.concat(
            [df for df in episode_results if not df.empty],
            ignore_index=True,
        )

        all_analyses.append(controller_data)
        summary_rows.append(
            summarize_controller(controller, controller_data)
        )

analysis_df = pd.concat(all_analyses, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)

analysis_df.to_csv(
    OUTPUT_DIR / "gamma_future_error_all_samples.csv",
    index=False,
)

summary_df.to_csv(
    OUTPUT_DIR / "gamma_future_error_summary.csv",
    index=False,
)

print("\nSummary:")
print(summary_df.to_string(index=False))

for window in FUTURE_WINDOWS:
    plt.figure(figsize=(9, 6))

    for controller in CONTROLLERS:
        selected = analysis_df[
            (analysis_df["controller"] == controller)
            & (analysis_df["future_window"] == window)
        ]

        # Plot a random subset so the chart is readable
        if len(selected) > 5000:
            selected = selected.sample(
                5000,
                random_state=0,
            )

        plt.scatter(
            selected["gamma_change"],
            selected["future_error_change"],
            alpha=0.15,
            s=8,
            label=controller,
        )

    plt.axhline(0.0, linestyle="--", linewidth=1)

    plt.xlabel(r"Gamma change: $\|\Gamma_t-\Gamma_{t-1}\|$")
    plt.ylabel(
        f"Formation error change after {window} steps"
    )
    plt.title(
        "Gamma Adjustment vs. Future Formation-Error Change"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"gamma_vs_future_error_{window}.png",
        dpi=180,
    )
    plt.close()