import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# Settings
# =====================================================
N_TRIALS = 10

SAMPLE_HZ = 50.0
DT = 1.0 / SAMPLE_HZ

PLATE_HALF = 0.150  # m, 300 x 300 mm plate

FORCE_MIN = 0.0
FORCE_MAX = 30.0

DURATION_MIN = 1.0
DURATION_MAX = 5.0

# OU force fluctuation model
OU_THETA = 2.5          # target force로 되돌아가는 강도
OU_SIGMA = 2.5          # 힘 흔들림 크기 [N]
OU_INITIAL_STD = 1.0    # 시작 힘의 랜덤 편차 [N]


# =====================================================
# Helpers
# =====================================================
def random_point(rng):
    return np.array([
        rng.uniform(-PLATE_HALF, PLATE_HALF),
        rng.uniform(-PLATE_HALF, PLATE_HALF),
        0.0,
    ], dtype=float)


def generate_ou_force_profile(target_force, n_samples, dt, rng):
    """
    시작부터 끝까지 힘이 0이 되지 않는 OU Process force profile.
    target_force 주변에서 자연스럽게 흔들림.
    """
    forces = np.zeros(n_samples, dtype=float)

    forces[0] = target_force + rng.normal(0.0, OU_INITIAL_STD)
    forces[0] = np.clip(forces[0], FORCE_MIN, FORCE_MAX)
    while forces[0] < 1e-5 :
        forces[0] = target_force + rng.normal(0.0, OU_INITIAL_STD)
        forces[0] = np.clip(forces[0], FORCE_MIN, FORCE_MAX)
        
    for k in range(1, n_samples):
        prev = forces[k - 1]

        dF = (
            OU_THETA * (target_force - prev) * dt
            + OU_SIGMA * np.sqrt(dt) * rng.normal()
        )

        forces[k] = prev + dF
        forces[k] = np.clip(forces[k], FORCE_MIN, FORCE_MAX)

    return forces


# =====================================================
# Main
# =====================================================
def main():
    seed = int(time.time_ns() % (2**32))
    rng = np.random.default_rng(seed)

    print(f"Random seed = {seed}")

    OUT_DIR = Path("./random_line_push_waypoints") / str(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_CSV = OUT_DIR / "random_line_waypoints.csv"
    OUT_META_CSV = OUT_DIR / "random_line_trial_meta.csv"

    waypoint_rows = []
    meta_rows = []

    for trial_idx in range(1, N_TRIALS + 1):
        p0 = random_point(rng)
        p1 = random_point(rng)

        force_target = rng.uniform(FORCE_MIN, FORCE_MAX)
        duration = rng.uniform(DURATION_MIN, DURATION_MAX)

        n_samples = int(math.ceil(duration / DT)) + 1
        times = np.linspace(0.0, duration, n_samples)

        forces = generate_ou_force_profile(
            target_force=force_target,
            n_samples=n_samples,
            dt=DT,
            rng=rng,
        )

        meta_rows.append({
            "seed": seed,
            "trial": trial_idx,
            "x0_m": p0[0],
            "y0_m": p0[1],
            "x1_m": p1[0],
            "y1_m": p1[1],
            "force_target_N": force_target,
            "force_mean_N": float(np.mean(forces)),
            "force_std_N": float(np.std(forces)),
            "force_min_N": float(np.min(forces)),
            "force_max_N": float(np.max(forces)),
            "duration_s": duration,
            "sample_hz": SAMPLE_HZ,
            "dt_s": DT,
            "samples": n_samples,
            "ou_theta": OU_THETA,
            "ou_sigma": OU_SIGMA,
            "ou_initial_std": OU_INITIAL_STD,
        })

        print(
            f"Trial {trial_idx:02d}: "
            f"({p0[0]*1000:.1f}, {p0[1]*1000:.1f}) mm -> "
            f"({p1[0]*1000:.1f}, {p1[1]*1000:.1f}) mm | "
            f"Ftarget={force_target:.2f} N | "
            f"Fmean={np.mean(forces):.2f} N | "
            f"T={duration:.2f} s | "
            f"samples={n_samples}"
        )

        for sample_idx, t in enumerate(times):
            u = 0.0 if duration <= 1e-12 else t / duration
            p = (1.0 - u) * p0 + u * p1

            current_force = forces[sample_idx]

            target_name = f"Line_T{trial_idx:02d}_{sample_idx:04d}"

            waypoint_rows.append({
                "seed": seed,
                "trial": trial_idx,
                "sample": sample_idx,
                "target_name": target_name,
                "time_s": t,
                "duration_s": duration,
                "x_m": p[0],
                "y_m": p[1],
                "z_m": p[2],
                "x_mm": p[0] * 1000.0,
                "y_mm": p[1] * 1000.0,
                "z_mm": p[2] * 1000.0,
                "force_target_N": force_target,
                "current_force_N": current_force,
            })

    df = pd.DataFrame(waypoint_rows)
    meta = pd.DataFrame(meta_rows)

    df.to_csv(OUT_CSV, index=False)
    meta.to_csv(OUT_META_CSV, index=False)

    print("")
    print("Saved:", OUT_CSV.resolve())
    print("Saved:", OUT_META_CSV.resolve())
    print("Total waypoints:", len(df))


if __name__ == "__main__":
    main()