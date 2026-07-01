import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# Input / output
# =====================================================
TARGET_CSV = "./robodk_press_targets_plane.csv"

OUT_DETAIL_CSV = "./auto/plate_offset_optimization_detail.csv"
OUT_SUMMARY_CSV = "./auto/plate_offset_optimization_summary.csv"
OUT_PLOT_ERROR = "./auto/plate_offset_error_curve.png"
OUT_PLOT_TRANSMISSION = "./auto/plate_offset_transmission_curve.png"


# =====================================================
# Plate / force settings
# =====================================================
PLATE_THICKNESS = 0.01
FORCE_N = 10.0
FORCE_PLATE = np.array([0.0, 0.0, -FORCE_N], dtype=float)

# offset search range [m]
OFFSET_MIN = 0.050
OFFSET_MAX = 0.150
OFFSET_STEP = 0.001

# realistic virtual sensor model
GAUSSIAN_SIGMA = 0.08
CONTACT_PATCH_RADIUS = 0.005
CONTACT_PATCH_POINTS = 16
TRANSMISSION_GAIN = 0.55

# deterministic tiny noise
USE_SENSOR_NOISE = True
FORCE_NOISE_STD = 0.005
TORQUE_NOISE_STD = 0.00002
BASE_NOISE_SEED = 42

# 5 sensors: 4 corner + 1 center
SENSOR_GAIN = np.array([1.00, 1.00, 1.00, 1.00, 1.00], dtype=float)


# =====================================================
# Layout
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


# =====================================================
# Virtual FT model
# =====================================================
def sample_contact_patch(center, radius=0.005, n=16):
    points = [center.copy()]

    for k in range(n):
        theta = 2.0 * np.pi * k / n
        points.append(center + np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            0.0,
        ]))

    return points


def compute_raw_influence(contact, sensors, sigma):
    diff = sensors - contact
    d = np.linalg.norm(diff, axis=1)
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def influence_to_efficiency(raw):
    s = np.sum(raw)
    eff = 1.0 - np.exp(-TRANSMISSION_GAIN * s)
    return float(np.clip(eff, 0.0, 1.0))


def apply_sensor_noise(ft_values, offset_idx, target_idx):
    if not USE_SENSOR_NOISE:
        return ft_values

    seed = BASE_NOISE_SEED + offset_idx * 100000 + target_idx
    rng = np.random.default_rng(seed)

    out = ft_values.copy()

    out[:, 0:3] += rng.normal(
        0.0,
        FORCE_NOISE_STD,
        size=out[:, 0:3].shape,
    )

    out[:, 3:6] += rng.normal(
        0.0,
        TORQUE_NOISE_STD,
        size=out[:, 3:6].shape,
    )

    return out


def compute_virtual_ft(contact, sensors, force, offset_idx, target_idx):
    n_sensors = len(sensors)
    ft_total = np.zeros((n_sensors, 7), dtype=float)

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
            ft_total[i, 6] += weights[i] / len(patch_points)

    ft_total = apply_sensor_noise(
        ft_total,
        offset_idx=offset_idx,
        target_idx=target_idx,
    )

    measured_force_sum = np.linalg.norm(np.sum(ft_total[:, 0:3], axis=0))
    transmission_efficiency = measured_force_sum / np.linalg.norm(force)

    return ft_total, measured_force_sum, transmission_efficiency


# =====================================================
# Contact estimation
# =====================================================
def estimate_contact_ls(ft_values, sensors):
    A = []
    b = []

    for i in range(len(sensors)):
        Fx, Fy, Fz = ft_values[i, 0:3]
        Tx, Ty, _ = ft_values[i, 3:6]
        sx, sy, sz = sensors[i]

        if abs(Fz) < 1e-9:
            continue

        # Tx = (y - sy)Fz - (0 - sz)Fy
        A.append([0.0, Fz])
        b.append(Tx + sy * Fz - sz * Fy)

        # Ty = (0 - sz)Fx - (x - sx)Fz
        A.append([-Fz, 0.0])
        b.append(Ty + sz * Fx - sx * Fz)

    if len(A) < 2:
        return np.zeros(3)

    xy, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return np.array([xy[0], xy[1], 0.0])


# =====================================================
# Main optimization
# =====================================================
def main():
    targets = pd.read_csv(TARGET_CSV)

    required = ["target_name", "x_m", "y_m", "z_m"]
    missing = [c for c in required if c not in targets.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    offsets = np.arange(
        OFFSET_MIN,
        OFFSET_MAX + OFFSET_STEP * 0.5,
        OFFSET_STEP,
    )

    detail_rows = []
    summary_rows = []

    for offset_idx, offset in enumerate(offsets):
        layout_name = f"offset_{int(round(offset * 1000)):03d}mm"
        sensors = square_center_layout(offset)

        errors = []
        transmissions = []
        measured_forces = []

        print(f"Testing {layout_name}")

        for target_idx, row in targets.iterrows():
            contact = np.array([
                float(row["x_m"]),
                float(row["y_m"]),
                0.0,
            ], dtype=float)

            ft_values, measured_force_sum, transmission_efficiency = compute_virtual_ft(
                contact=contact,
                sensors=sensors,
                force=FORCE_PLATE,
                offset_idx=offset_idx,
                target_idx=target_idx,
            )

            est = estimate_contact_ls(ft_values, sensors)
            error_mm = np.linalg.norm(est[:2] - contact[:2]) * 1000.0

            errors.append(error_mm)
            transmissions.append(transmission_efficiency)
            measured_forces.append(measured_force_sum)

            detail_rows.append({
                "layout": layout_name,
                "offset_m": offset,
                "offset_mm": offset * 1000.0,
                "target_name": row["target_name"],
                "contact_x_m": contact[0],
                "contact_y_m": contact[1],
                "est_x_m": est[0],
                "est_y_m": est[1],
                "error_mm": error_mm,
                "applied_force_N": FORCE_N,
                "measured_force_sum_N": measured_force_sum,
                "transmission_efficiency": transmission_efficiency,
            })

        errors = np.array(errors, dtype=float)
        transmissions = np.array(transmissions, dtype=float)
        measured_forces = np.array(measured_forces, dtype=float)

        summary_rows.append({
            "layout": layout_name,
            "offset_m": offset,
            "offset_mm": offset * 1000.0,
            "targets": len(errors),
            "mean_error_mm": float(np.mean(errors)),
            "rmse_error_mm": float(math.sqrt(np.mean(errors ** 2))),
            "max_error_mm": float(np.max(errors)),
            "mean_transmission_efficiency": float(np.mean(transmissions)),
            "min_transmission_efficiency": float(np.min(transmissions)),
            "mean_measured_force_N": float(np.mean(measured_forces)),
        })

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)

    detail.to_csv(OUT_DETAIL_CSV, index=False)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    best_mean = summary.sort_values("mean_error_mm").iloc[0]
    best_max = summary.sort_values("max_error_mm").iloc[0]
    best_trans = summary.sort_values("mean_transmission_efficiency", ascending=False).iloc[0]

    print("")
    print("=" * 70)
    print("Optimization result")
    print("=" * 70)
    print(
        f"Best mean error: {best_mean['layout']} | "
        f"{best_mean['mean_error_mm']:.4f} mm"
    )
    print(
        f"Best max error: {best_max['layout']} | "
        f"{best_max['max_error_mm']:.4f} mm"
    )
    print(
        f"Best transmission: {best_trans['layout']} | "
        f"{best_trans['mean_transmission_efficiency'] * 100:.2f} %"
    )

    print("")
    print("Saved:", Path(OUT_DETAIL_CSV).resolve())
    print("Saved:", Path(OUT_SUMMARY_CSV).resolve())

    # =====================================================
    # Plot 1: error vs offset
    # =====================================================
    plt.figure(figsize=(9, 5))
    plt.plot(summary["offset_mm"], summary["mean_error_mm"], marker="o", label="Mean error")
    plt.plot(summary["offset_mm"], summary["rmse_error_mm"], marker="o", label="RMSE")
    plt.plot(summary["offset_mm"], summary["max_error_mm"], marker="o", label="Max error")
    plt.axvline(best_mean["offset_mm"], linestyle="--", label=f"Best mean: {best_mean['offset_mm']:.0f} mm")
    plt.xlabel("Corner sensor offset [mm]")
    plt.ylabel("Contact estimation error [mm]")
    plt.title("Sensor offset optimization: contact estimation error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT_ERROR, dpi=200)
    print("Saved:", Path(OUT_PLOT_ERROR).resolve())

    # =====================================================
    # Plot 2: transmission vs offset
    # =====================================================
    plt.figure(figsize=(9, 5))
    plt.plot(
        summary["offset_mm"],
        summary["mean_transmission_efficiency"] * 100.0,
        marker="o",
        label="Mean transmission",
    )
    plt.plot(
        summary["offset_mm"],
        summary["min_transmission_efficiency"] * 100.0,
        marker="o",
        label="Min transmission",
    )
    plt.axvline(best_trans["offset_mm"], linestyle="--", label=f"Best transmission: {best_trans['offset_mm']:.0f} mm")
    plt.xlabel("Corner sensor offset [mm]")
    plt.ylabel("Transmission efficiency [%]")
    plt.title("Sensor offset optimization: transmission efficiency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT_TRANSMISSION, dpi=200)
    print("Saved:", Path(OUT_PLOT_TRANSMISSION).resolve())


if __name__ == "__main__":
    main()