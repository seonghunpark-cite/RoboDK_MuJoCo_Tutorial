import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# Settings
# =====================================================
OUT_DIR = Path("./random_line_push_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "random_line_push_5ft.csv"
OUT_META_CSV = OUT_DIR / "random_line_push_trial_meta.csv"

N_TRIALS = 10
SAMPLE_HZ = 50.0
DT = 1.0 / SAMPLE_HZ

PLATE_HALF = 0.150          # 300 x 300 mm plate, m
PLATE_THICKNESS = 0.01      # m

FORCE_MIN = 10.0
FORCE_MAX = 30.0

DURATION_MIN = 1.0
DURATION_MAX = 5.0

RAMP_TIME = 0.2             # sec, absolute ramp up/down time
RANDOM_SEED = 42

# Optimized layout: 4 corner sensors at 93 mm + 1 center sensor
SENSOR_OFFSET = 0.093       # m

# Realistic transmission model
GAUSSIAN_SIGMA = 0.08
CONTACT_PATCH_RADIUS = 0.005
CONTACT_PATCH_POINTS = 16
TRANSMISSION_GAIN = 0.55

# Tiny deterministic sensor noise
USE_SENSOR_NOISE = True
FORCE_NOISE_STD = 0.005
TORQUE_NOISE_STD = 0.00002
BASE_NOISE_SEED = 1000

SENSOR_GAIN = np.array([1.00, 0.99, 1.01, 1.00, 1.00], dtype=float)


# =====================================================
# Sensor layout
# =====================================================
def square_center_layout(offset):
    z = -PLATE_THICKNESS
    return np.array([
        [-offset, -offset, z],   # FT1
        [ offset, -offset, z],   # FT2
        [ offset,  offset, z],   # FT3
        [-offset,  offset, z],   # FT4
        [ 0.0,     0.0,    z],   # FT5 center
    ], dtype=float)


SENSORS = square_center_layout(SENSOR_OFFSET)


# =====================================================
# Force profile
# =====================================================
def force_profile(t, duration, target_force):
    """
    0~0.2 s: ramp up
    middle: constant force
    last 0.2 s: ramp down
    """
    ramp = min(RAMP_TIME, duration * 0.5)

    if t <= ramp:
        return target_force * (t / ramp)

    if t >= duration - ramp:
        return target_force * ((duration - t) / ramp)

    return target_force


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


def apply_sensor_noise(ft_values, trial_idx, sample_idx):
    if not USE_SENSOR_NOISE:
        return ft_values

    seed = BASE_NOISE_SEED + trial_idx * 100000 + sample_idx
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


def compute_virtual_ft(contact, sensors, force, trial_idx, sample_idx):
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
        trial_idx=trial_idx,
        sample_idx=sample_idx,
    )

    applied_force_N = np.linalg.norm(force)
    measured_force_sum_N = np.linalg.norm(np.sum(ft_total[:, 0:3], axis=0))

    if applied_force_N > 1e-12:
        transmission_efficiency = measured_force_sum_N / applied_force_N
    else:
        transmission_efficiency = 0.0

    return ft_total, measured_force_sum_N, transmission_efficiency

def plot_single_sensor(trial_df, trial_idx, sensor_id):
    sub = trial_df[trial_df["sensor_id"] == sensor_id]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 7),
        sharex=True,
    )

    signals = [
        ("Fx_N", "Fx [N]"),
        ("Fy_N", "Fy [N]"),
        ("Fz_N", "Fz [N]"),
        ("Tx_Nm", "Tx [N·m]"),
        ("Ty_Nm", "Ty [N·m]"),
        ("Tz_Nm", "Tz [N·m]"),
    ]

    for ax, (col, ylabel) in zip(axes.flatten(), signals):
        ax.plot(
            sub["time_s"],
            sub[col],
            linewidth=2,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True)

    for ax in axes[1, :]:
        ax.set_xlabel("Time [s]")

    fig.suptitle(
        f"Trial {trial_idx:02d} | FT{sensor_id}",
        fontsize=14,
    )

    plt.tight_layout()

    out_path = OUT_DIR / f"trial_{trial_idx:02d}_FT{sensor_id}_6axis.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_trial_all_sensors_fz(trial_df, trial_idx):
    plt.figure(figsize=(9, 5))

    for sensor_id in range(1, 6):
        sub = trial_df[trial_df["sensor_id"] == sensor_id]
        plt.plot(
            sub["time_s"],
            sub["Fz_N"],
            linewidth=2,
            label=f"FT{sensor_id}",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Fz [N]")
    plt.title(f"Trial {trial_idx:02d} | All sensors Fz")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / f"trial_{trial_idx:02d}_all_sensors_Fz.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

# =====================================================
# Trial generation
# =====================================================
def random_point(rng):
    return np.array([
        rng.uniform(-PLATE_HALF, PLATE_HALF),
        rng.uniform(-PLATE_HALF, PLATE_HALF),
        0.0,
    ], dtype=float)


def generate_trials():
    rng = np.random.default_rng(RANDOM_SEED)
    trials = []

    for trial_idx in range(1, N_TRIALS + 1):
        p0 = random_point(rng)
        p1 = random_point(rng)

        force_N = rng.uniform(FORCE_MIN, FORCE_MAX)
        duration = rng.uniform(DURATION_MIN, DURATION_MAX)

        trials.append({
            "trial": trial_idx,
            "x0_m": p0[0],
            "y0_m": p0[1],
            "x1_m": p1[0],
            "y1_m": p1[1],
            "force_N": force_N,
            "duration_s": duration,
        })

    return trials


# =====================================================
# Plotting
# =====================================================

def plot_all_trials(df):
    for trial_idx in range(1, N_TRIALS + 1):
        trial_df = df[df["trial"] == trial_idx]

        for sensor_id in range(1, 6):
            plot_single_sensor(
                trial_df=trial_df,
                trial_idx=trial_idx,
                sensor_id=sensor_id,
            )

        plot_trial_all_sensors_fz(
            trial_df=trial_df,
            trial_idx=trial_idx,
        )


# =====================================================
# Main
# =====================================================
def main():
    trials = generate_trials()

    all_rows = []
    meta_rows = []

    for trial in trials:
        trial_idx = trial["trial"]

        p0 = np.array([trial["x0_m"], trial["y0_m"], 0.0], dtype=float)
        p1 = np.array([trial["x1_m"], trial["y1_m"], 0.0], dtype=float)

        force_target_N = trial["force_N"]
        duration = trial["duration_s"]

        n_samples = int(math.ceil(duration / DT)) + 1
        times = np.linspace(0.0, duration, n_samples)

        meta_rows.append({
            "trial": trial_idx,
            "x0_m": p0[0],
            "y0_m": p0[1],
            "x1_m": p1[0],
            "y1_m": p1[1],
            "force_target_N": force_target_N,
            "duration_s": duration,
            "samples": n_samples,
            "ramp_time_s": RAMP_TIME,
        })

        print(
            f"Trial {trial_idx:02d}: "
            f"({p0[0]*1000:.1f}, {p0[1]*1000:.1f}) mm -> "
            f"({p1[0]*1000:.1f}, {p1[1]*1000:.1f}) mm | "
            f"F={force_target_N:.2f} N | "
            f"T={duration:.2f} s"
        )

        for sample_idx, t in enumerate(times):
            u = 0.0 if duration <= 1e-12 else t / duration
            contact = (1.0 - u) * p0 + u * p1

            current_force_N = force_profile(
                t=t,
                duration=duration,
                target_force=force_target_N,
            )

            force = np.array([0.0, 0.0, -current_force_N], dtype=float)

            ft_values, measured_force_sum_N, transmission_efficiency = compute_virtual_ft(
                contact=contact,
                sensors=SENSORS,
                force=force,
                trial_idx=trial_idx,
                sample_idx=sample_idx,
            )

            for sensor_idx in range(5):
                Fx, Fy, Fz = ft_values[sensor_idx, 0:3]
                Tx, Ty, Tz = ft_values[sensor_idx, 3:6]
                weight = ft_values[sensor_idx, 6]

                all_rows.append({
                    "trial": trial_idx,
                    "sample": sample_idx,
                    "time_s": t,
                    "duration_s": duration,
                    "ramp_time_s": RAMP_TIME,
                    "contact_x_m": contact[0],
                    "contact_y_m": contact[1],
                    "sensor_id": sensor_idx + 1,
                    "weight": weight,
                    "force_target_N": force_target_N,
                    "current_force_N": current_force_N,
                    "Fx_N": Fx,
                    "Fy_N": Fy,
                    "Fz_N": Fz,
                    "Tx_Nm": Tx,
                    "Ty_Nm": Ty,
                    "Tz_Nm": Tz,
                    "measured_force_sum_N": measured_force_sum_N,
                    "transmission_efficiency": transmission_efficiency,
                })

    df = pd.DataFrame(all_rows)
    meta = pd.DataFrame(meta_rows)

    df.to_csv(OUT_CSV, index=False)
    meta.to_csv(OUT_META_CSV, index=False)

    print("")
    print("Saved:", OUT_CSV.resolve())
    print("Saved:", OUT_META_CSV.resolve())

    plot_all_trials(df)

    print("")
    print(f"Saved 50 plots to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()