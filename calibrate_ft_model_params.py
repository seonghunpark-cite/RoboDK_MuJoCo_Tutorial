import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# =====================================================
# Settings
# =====================================================

SEED = "AUTO_LATEST"
BASE_DIR = Path("./random_line_push_waypoints")

# If SEED == "AUTO_LATEST", use latest seed folder
# Otherwise use BASE_DIR / SEED
INPUT_CSV_NAME = "mujoco_random_line_ft.csv"
OUT_JSON_NAME = "calibrated_ft_params.json"
OUT_SUMMARY_CSV_NAME = "calibration_fit_summary.csv"

PLATE_THICKNESS = 0.01
SENSOR_OFFSET = 0.093

CONTACT_PATCH_RADIUS = 0.005
CONTACT_PATCH_POINTS = 16

MIN_TRUE_FORCE_N = 0.2

# Initial guess
INIT_SIGMA = 0.08
INIT_TRANSMISSION_GAIN = 0.55
INIT_SENSOR_GAIN = np.ones(5)

# Bounds
SIGMA_MIN = 0.02
SIGMA_MAX = 0.20

TRANSMISSION_GAIN_MIN = 0.01
TRANSMISSION_GAIN_MAX = 5.0

SENSOR_GAIN_MIN = 0.5
SENSOR_GAIN_MAX = 1.5

TORQUE_RES_SCALE = 50.0


# =====================================================
# Path helpers
# =====================================================
def find_experiment_dir():
    if SEED != "AUTO_LATEST":
        return BASE_DIR / str(SEED)

    seed_dirs = [
        p for p in BASE_DIR.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]

    if not seed_dirs:
        raise RuntimeError(f"No seed folders found in {BASE_DIR}")

    return max(seed_dirs, key=lambda p: p.stat().st_mtime)


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


def influence_to_efficiency(raw, transmission_gain):
    s = np.sum(raw)
    eff = 1.0 - np.exp(-transmission_gain * s)
    return float(np.clip(eff, 0.0, 1.0))


def compute_virtual_ft(
    contact,
    sensors,
    force,
    sigma,
    transmission_gain,
    sensor_gain,
):
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
            sigma=sigma,
        )

        efficiency = influence_to_efficiency(
            raw=raw,
            transmission_gain=transmission_gain,
        )

        if np.sum(raw) < 1e-12:
            weights = np.ones(n_sensors) / n_sensors
        else:
            weights = raw / np.sum(raw)

        patch_force = force / len(patch_points)
        measured_patch_force = efficiency * patch_force

        for i in range(n_sensors):
            Fi = weights[i] * measured_patch_force
            Fi = sensor_gain[i] * Fi

            ri = p - sensors[i]
            Ti = np.cross(ri, Fi)

            ft_total[i, 0:3] += Fi
            ft_total[i, 3:6] += Ti

    return ft_total


# =====================================================
# Parameter transform
# =====================================================
def pack_initial_params():
    return np.array([
        math.log(INIT_SIGMA),
        math.log(INIT_TRANSMISSION_GAIN),
        *np.log(INIT_SENSOR_GAIN),
    ], dtype=float)


def unpack_params(p):
    sigma = float(np.exp(p[0]))
    transmission_gain = float(np.exp(p[1]))

    sensor_gain = np.exp(p[2:7])

    return sigma, transmission_gain, sensor_gain


def bounds():
    lb = np.array([
        math.log(SIGMA_MIN),
        math.log(TRANSMISSION_GAIN_MIN),
        *([math.log(SENSOR_GAIN_MIN)] * 5),
    ], dtype=float)

    ub = np.array([
        math.log(SIGMA_MAX),
        math.log(TRANSMISSION_GAIN_MAX),
        *([math.log(SENSOR_GAIN_MAX)] * 5),
    ], dtype=float)

    return lb, ub


# =====================================================
# Calibration data
# =====================================================
def build_calibration_samples(df):
    samples = []

    grouped = df.groupby(["trial", "sample"])

    for (trial, sample), g in grouped:
        g = g.sort_values("sensor_id")

        if len(g) != 5:
            continue

        true_force = float(g["current_force_N"].iloc[0])

        if true_force < MIN_TRUE_FORCE_N:
            continue

        contact = np.array([
            float(g["contact_x_m"].iloc[0]),
            float(g["contact_y_m"].iloc[0]),
            0.0,
        ], dtype=float)

        force = np.array([0.0, 0.0, -true_force], dtype=float)

        measured_ft = g[[
            "Fx_N", "Fy_N", "Fz_N",
            "Tx_Nm", "Ty_Nm", "Tz_Nm",
        ]].to_numpy(dtype=float)

        samples.append({
            "trial": int(trial),
            "sample": int(sample),
            "time_s": float(g["time_s"].iloc[0]),
            "contact": contact,
            "force": force,
            "true_force_N": true_force,
            "measured_ft": measured_ft,
        })

    return samples


# =====================================================
# Optimization
# =====================================================
def residual_all(params, samples):
    sigma, transmission_gain, sensor_gain = unpack_params(params)

    residuals = []

    for s in samples:
        pred = compute_virtual_ft(
            contact=s["contact"],
            sensors=SENSORS,
            force=s["force"],
            sigma=sigma,
            transmission_gain=transmission_gain,
            sensor_gain=sensor_gain,
        )

        measured = s["measured_ft"]

        force_res = pred[:, 0:3] - measured[:, 0:3]
        torque_res = (pred[:, 3:6] - measured[:, 3:6]) * TORQUE_RES_SCALE

        residuals.extend(force_res.reshape(-1))
        residuals.extend(torque_res.reshape(-1))

    return np.array(residuals, dtype=float)


def compute_fit_summary(samples, sigma, transmission_gain, sensor_gain):
    rows = []

    for s in samples:
        pred = compute_virtual_ft(
            contact=s["contact"],
            sensors=SENSORS,
            force=s["force"],
            sigma=sigma,
            transmission_gain=transmission_gain,
            sensor_gain=sensor_gain,
        )

        measured = s["measured_ft"]

        force_rmse = float(np.sqrt(np.mean((pred[:, 0:3] - measured[:, 0:3]) ** 2)))
        torque_rmse = float(np.sqrt(np.mean((pred[:, 3:6] - measured[:, 3:6]) ** 2)))

        measured_force_sum = float(np.linalg.norm(np.sum(measured[:, 0:3], axis=0)))
        predicted_force_sum = float(np.linalg.norm(np.sum(pred[:, 0:3], axis=0)))

        rows.append({
            "trial": s["trial"],
            "sample": s["sample"],
            "time_s": s["time_s"],
            "true_force_N": s["true_force_N"],
            "measured_force_sum_N": measured_force_sum,
            "predicted_force_sum_N": predicted_force_sum,
            "force_rmse_N": force_rmse,
            "torque_rmse_Nm": torque_rmse,
        })

    return pd.DataFrame(rows)


# =====================================================
# Main
# =====================================================
def main():
    exp_dir = find_experiment_dir()

    in_csv = exp_dir /"mujoco_random_line_results"/ INPUT_CSV_NAME
    out_json = exp_dir / OUT_JSON_NAME
    out_summary_csv = exp_dir / OUT_SUMMARY_CSV_NAME

    if not in_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {in_csv}\n"
            f"Copy mujoco_random_line_ft.csv into this seed folder first."
        )

    print("Experiment dir:", exp_dir.resolve())
    print("Input CSV:", in_csv.resolve())

    df = pd.read_csv(in_csv)

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

    samples = build_calibration_samples(df)

    if not samples:
        raise RuntimeError("No valid calibration samples.")

    print("Calibration samples:", len(samples))

    x0 = pack_initial_params()
    lb, ub = bounds()

    result = least_squares(
        residual_all,
        x0=x0,
        bounds=(lb, ub),
        args=(samples,),
        max_nfev=300,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        verbose=1,
    )

    sigma, transmission_gain, sensor_gain = unpack_params(result.x)

    summary_df = compute_fit_summary(
        samples=samples,
        sigma=sigma,
        transmission_gain=transmission_gain,
        sensor_gain=sensor_gain,
    )

    summary_df.to_csv(out_summary_csv, index=False)

    output = {
        "model": "gaussian_global_transmission",
        "source_csv": str(in_csv),
        "calibration_samples": len(samples),

        "parameters": {
            "gaussian_sigma_m": sigma,
            "transmission_gain": transmission_gain,
            "sensor_gain": {
                "FT1": float(sensor_gain[0]),
                "FT2": float(sensor_gain[1]),
                "FT3": float(sensor_gain[2]),
                "FT4": float(sensor_gain[3]),
                "FT5": float(sensor_gain[4]),
            },
        },

        "settings": {
            "sensor_offset_m": SENSOR_OFFSET,
            "plate_thickness_m": PLATE_THICKNESS,
            "contact_patch_radius_m": CONTACT_PATCH_RADIUS,
            "contact_patch_points": CONTACT_PATCH_POINTS,
            "min_true_force_N": MIN_TRUE_FORCE_N,
            "torque_res_scale": TORQUE_RES_SCALE,
            "sigma_bounds_m": [SIGMA_MIN, SIGMA_MAX],
            "transmission_gain_bounds": [
                TRANSMISSION_GAIN_MIN,
                TRANSMISSION_GAIN_MAX,
            ],
            "sensor_gain_bounds": [
                SENSOR_GAIN_MIN,
                SENSOR_GAIN_MAX,
            ],
        },

        "fit_quality": {
            "optimizer_cost": float(result.cost),
            "optimizer_success": bool(result.success),
            "optimizer_message": result.message,
            "mean_force_rmse_N": float(summary_df["force_rmse_N"].mean()),
            "mean_torque_rmse_Nm": float(summary_df["torque_rmse_Nm"].mean()),
            "mean_measured_force_sum_N": float(summary_df["measured_force_sum_N"].mean()),
            "mean_predicted_force_sum_N": float(summary_df["predicted_force_sum_N"].mean()),
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("")
    print("Calibrated parameters")
    print("-" * 60)
    print(f"gaussian_sigma_m   = {sigma:.6f}")
    print(f"transmission_gain  = {transmission_gain:.6f}")
    print(f"sensor_gain        = {sensor_gain}")
    print("")
    print("Saved:", out_json.resolve())
    print("Saved:", out_summary_csv.resolve())


if __name__ == "__main__":
    main()