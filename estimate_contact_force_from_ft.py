import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# =====================================================
# Paths
# =====================================================
IN_CSV = "./random_line_push_waypoints/3666223592/mujoco_random_line_results/mujoco_random_line_ft.csv"

OUT_DIR = Path("./random_line_push_waypoints/3666223592/estimated_contact_force_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "estimated_contact_force.csv"


# =====================================================
# Plate / sensor settings
# =====================================================
PLATE_HALF = 0.150
PLATE_THICKNESS = 0.01
SENSOR_OFFSET = 0.093

GAUSSIAN_SIGMA = 0.08
CONTACT_PATCH_RADIUS = 0.005
CONTACT_PATCH_POINTS = 16
TRANSMISSION_GAIN = 0.55

SENSOR_GAIN = np.array([1.00, 1.00, 1.00, 1.00, 1.00], dtype=float)

FORCE_MIN = 0.0
FORCE_MAX = 35.0


# =====================================================
# Sensor layout
# =====================================================
def square_center_layout(offset):
    z = -PLATE_THICKNESS
    return np.array([
        [-offset, -offset, z],
        [ offset, -offset, z],
        [ offset,  offset, z],
        [-offset,  offset, z],
        [ 0.0,     0.0,    z],
    ], dtype=float)


SENSORS = square_center_layout(SENSOR_OFFSET)


# =====================================================
# Forward model
# =====================================================
def sample_contact_patch(center, radius=0.005, n=16):
    points = [center.copy()]

    for k in range(n):
        theta = 2.0 * np.pi * k / n
        points.append(center + np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            0.0,
        ], dtype=float))

    return points


def compute_raw_influence(contact, sensors, sigma):
    diff = sensors - contact
    d = np.linalg.norm(diff, axis=1)
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def influence_to_efficiency(raw):
    s = np.sum(raw)
    eff = 1.0 - np.exp(-TRANSMISSION_GAIN * s)
    return float(np.clip(eff, 0.0, 1.0))


def compute_virtual_ft_no_noise(contact, sensors, force):
    n_sensors = len(sensors)
    ft_total = np.zeros((n_sensors, 6), dtype=float)

    patch_points = sample_contact_patch(
        contact,
        radius=CONTACT_PATCH_RADIUS,
        n=CONTACT_PATCH_POINTS,
    )

    for p in patch_points:
        raw = compute_raw_influence(
            contact=p,
            sensors=sensors,
            sigma=GAUSSIAN_SIGMA,
        )

        efficiency = influence_to_efficiency(raw)

        if np.sum(raw) < 1e-12:
            weights = np.ones(n_sensors) / n_sensors
        else:
            weights = raw / np.sum(raw)

        patch_force = force / len(patch_points)
        measured_patch_force = efficiency * patch_force

        for i in range(n_sensors):
            Fi = weights[i] * measured_patch_force
            Fi = SENSOR_GAIN[i] * Fi

            ri = p - sensors[i]
            Ti = np.cross(ri, Fi)

            ft_total[i, 0:3] += Fi
            ft_total[i, 3:6] += Ti

    return ft_total


# =====================================================
# Estimation
# =====================================================
def measured_matrix(group):
    group = group.sort_values("sensor_id")

    return group[[
        "Fx_N", "Fy_N", "Fz_N",
        "Tx_Nm", "Ty_Nm", "Tz_Nm",
    ]].to_numpy(dtype=float)


def residual(params, measured_ft):
    x, y, force_n = params

    contact = np.array([x, y, 0.0], dtype=float)
    force = np.array([0.0, 0.0, -force_n], dtype=float)

    pred_ft = compute_virtual_ft_no_noise(
        contact=contact,
        sensors=SENSORS,
        force=force,
    )

    # force and torque have different units/scales.
    # torque scale is amplified so it contributes to the optimization.
    force_res = pred_ft[:, 0:3] - measured_ft[:, 0:3]
    torque_res = (pred_ft[:, 3:6] - measured_ft[:, 3:6]) * 50.0

    return np.concatenate([
        force_res.reshape(-1),
        torque_res.reshape(-1),
    ])


def estimate_one_sample(group, prev_guess=None):
    measured_ft = measured_matrix(group)

    if prev_guess is None:
        # Initial force estimate from measured resultant.
        f0 = abs(float(np.sum(measured_ft[:, 2])))
        f0 = np.clip(f0, 1.0, FORCE_MAX)

        # Initial position estimate from Fz-weighted centroid.
        fz_abs = np.abs(measured_ft[:, 2])
        if np.sum(fz_abs) > 1e-12:
            xy0 = np.sum(SENSORS[:, :2] * fz_abs[:, None], axis=0) / np.sum(fz_abs)
        else:
            xy0 = np.array([0.0, 0.0])

        x0 = np.array([xy0[0], xy0[1], f0], dtype=float)
    else:
        x0 = prev_guess.copy()

    lb = np.array([-PLATE_HALF, -PLATE_HALF, FORCE_MIN], dtype=float)
    ub = np.array([ PLATE_HALF,  PLATE_HALF, FORCE_MAX], dtype=float)

    result = least_squares(
        residual,
        x0=x0,
        bounds=(lb, ub),
        args=(measured_ft,),
        max_nfev=80,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )

    return result.x, result.cost, result.success


# =====================================================
# Plotting
# =====================================================
def plot_trial_trajectory(df_trial, trial_idx):
    plt.figure(figsize=(6, 6))

    plt.plot(
        df_trial["true_x_m"] * 1000,
        df_trial["true_y_m"] * 1000,
        linewidth=2,
        label="True trajectory",
    )

    plt.plot(
        df_trial["est_x_m"] * 1000,
        df_trial["est_y_m"] * 1000,
        linewidth=2,
        linestyle="--",
        label="Estimated trajectory",
    )

    plt.scatter(
        df_trial["true_x_m"].iloc[0] * 1000,
        df_trial["true_y_m"].iloc[0] * 1000,
        s=50,
        label="Start",
    )

    plt.scatter(
        df_trial["true_x_m"].iloc[-1] * 1000,
        df_trial["true_y_m"].iloc[-1] * 1000,
        s=50,
        label="End",
    )

    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title(f"Trial {trial_idx:02d} contact trajectory")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_DIR / f"trial_{trial_idx:02d}_trajectory_compare.png", dpi=200)
    plt.close()


def plot_trial_position_error(df_trial, trial_idx):
    plt.figure(figsize=(9, 4.5))

    plt.plot(
        df_trial["time_s"],
        df_trial["position_error_mm"],
        linewidth=2,
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Position error [mm]")
    plt.title(f"Trial {trial_idx:02d} position estimation error")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(OUT_DIR / f"trial_{trial_idx:02d}_position_error.png", dpi=200)
    plt.close()


def plot_trial_force_compare(df_trial, trial_idx):
    plt.figure(figsize=(9, 4.5))

    plt.plot(
        df_trial["time_s"],
        df_trial["true_force_N"],
        linewidth=2,
        label="True force",
    )

    plt.plot(
        df_trial["time_s"],
        df_trial["est_force_N"],
        linewidth=2,
        linestyle="--",
        label="Estimated force",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title(f"Trial {trial_idx:02d} force comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_DIR / f"trial_{trial_idx:02d}_force_compare.png", dpi=200)
    plt.close()


def plot_trial_force_error(df_trial, trial_idx):
    plt.figure(figsize=(9, 4.5))

    plt.plot(
        df_trial["time_s"],
        df_trial["force_error_N"],
        linewidth=2,
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Force error [N]")
    plt.title(f"Trial {trial_idx:02d} force estimation error")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(OUT_DIR / f"trial_{trial_idx:02d}_force_error.png", dpi=200)
    plt.close()


def plot_summary(result_df):
    summary = (
        result_df
        .groupby("trial")
        .agg(
            mean_position_error_mm=("position_error_mm", "mean"),
            max_position_error_mm=("position_error_mm", "max"),
            mean_force_error_N=("force_error_N", "mean"),
            max_force_error_N=("force_error_N", "max"),
        )
        .reset_index()
    )

    summary.to_csv(OUT_DIR / "estimation_summary.csv", index=False)

    plt.figure(figsize=(9, 4.5))
    plt.bar(summary["trial"], summary["mean_position_error_mm"])
    plt.xlabel("Trial")
    plt.ylabel("Mean position error [mm]")
    plt.title("Mean contact position error by trial")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "summary_mean_position_error.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.bar(summary["trial"], summary["mean_force_error_N"])
    plt.xlabel("Trial")
    plt.ylabel("Mean force error [N]")
    plt.title("Mean force estimation error by trial")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "summary_mean_force_error.png", dpi=200)
    plt.close()

    print("")
    print("Summary")
    print(summary)


# =====================================================
# Main
# =====================================================
def main():
    df = pd.read_csv(IN_CSV)

    required = [
        "trial", "sample", "time_s",
        "contact_x_m", "contact_y_m",
        "current_force_N",
        "sensor_id",
        "Fx_N", "Fy_N", "Fz_N",
        "Tx_Nm", "Ty_Nm", "Tz_Nm",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    result_rows = []

    for trial_idx, trial_df in df.groupby("trial"):
        trial_df = trial_df.sort_values(["sample", "sensor_id"])

        prev_guess = None

        print(f"Estimating Trial {int(trial_idx):02d}")

        for sample_idx, sample_df in trial_df.groupby("sample"):
            sample_df = sample_df.sort_values("sensor_id")

            if len(sample_df) != 5:
                continue

            est, cost, success = estimate_one_sample(
                sample_df,
                prev_guess=prev_guess,
            )

            prev_guess = est

            true_x = float(sample_df["contact_x_m"].iloc[0])
            true_y = float(sample_df["contact_y_m"].iloc[0])
            true_force = float(sample_df["current_force_N"].iloc[0])

            est_x, est_y, est_force = est

            position_error_mm = math.sqrt(
                (est_x - true_x) ** 2 +
                (est_y - true_y) ** 2
            ) * 1000.0

            force_error_N = abs(est_force - true_force)

            result_rows.append({
                "trial": int(trial_idx),
                "sample": int(sample_idx),
                "time_s": float(sample_df["time_s"].iloc[0]),

                "true_x_m": true_x,
                "true_y_m": true_y,
                "true_force_N": true_force,

                "est_x_m": est_x,
                "est_y_m": est_y,
                "est_force_N": est_force,

                "position_error_mm": position_error_mm,
                "force_error_N": force_error_N,

                "optimizer_cost": cost,
                "optimizer_success": success,
            })

    result = pd.DataFrame(result_rows)
    result.to_csv(OUT_CSV, index=False)

    print("")
    print("Saved:", OUT_CSV.resolve())

    for trial_idx, trial_result in result.groupby("trial"):
        plot_trial_trajectory(trial_result, int(trial_idx))
        plot_trial_position_error(trial_result, int(trial_idx))
        plot_trial_force_compare(trial_result, int(trial_idx))
        plot_trial_force_error(trial_result, int(trial_idx))

    plot_summary(result)

    print("")
    print("Saved plots to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()