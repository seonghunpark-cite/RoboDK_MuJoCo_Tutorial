import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# Paths / settings
# =====================================================
MODEL_PATH = "./assets/fairino_description/urdf/fairino5_v6_converted_plate.xml"
CSV_PATH = "./random_line_push_waypoints/82093048/random_line_robodk_joints.csv"
OUT_DIR = Path("./random_line_push_waypoints/82093048/mujoco_random_line_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "mujoco_random_line_ft.csv"


TARGET_HOLD_SEC = 0.02   # 50 Hz
TRIAL_PAUSE_SEC = 0.5

PLATE_WORLD_CENTER = np.array([0.45, 0.0, 0.05])
PLATE_TO_WORLD_R = np.eye(3)

PLATE_THICKNESS = 0.01
SENSOR_OFFSET = 0.093

GAUSSIAN_SIGMA = 0.08
CONTACT_PATCH_RADIUS = 0.005
CONTACT_PATCH_POINTS = 16
TRANSMISSION_GAIN = 0.55

USE_SENSOR_NOISE = True
FORCE_NOISE_STD = 0.005
TORQUE_NOISE_STD = 0.00002
BASE_NOISE_SEED = 1000

SENSOR_GAIN = np.array([1.00, 1.00, 1.00, 1.00, 1.00], dtype=float)


# =====================================================
# Sensors: 4 corner + 1 center
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
# Helpers
# =====================================================
def read_csv_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = [
        "trial", "sample", "target_name", "time_s",
        "x_m", "y_m", "z_m",
        "current_force_N",
        "j1_rad", "j2_rad", "j3_rad", "j4_rad", "j5_rad", "j6_rad",
    ]

    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    return rows


def f(row, key):
    return float(row[key])


def plate_to_world_point(p):
    return PLATE_WORLD_CENTER + PLATE_TO_WORLD_R @ p


def plate_to_world_vec(v):
    return PLATE_TO_WORLD_R @ v


def set_robot_q(model, data, q):
    for k, name in enumerate(["j1", "j2", "j3", "j4", "j5", "j6"]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint not found: {name}")
        data.qpos[model.jnt_qposadr[jid]] = q[k]


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return

    idx = scene.ngeom
    mujoco.mjv_initGeom(
        scene.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0]),
        pos=np.asarray(pos, dtype=float),
        mat=np.eye(3).reshape(-1),
        rgba=np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def add_capsule(scene, p1, p2, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return

    idx = scene.ngeom
    mujoco.mjv_initGeom(
        scene.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).reshape(-1),
        rgba=np.asarray(rgba, dtype=float),
    )

    mujoco.mjv_connector(
        scene.geoms[idx],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(p1, dtype=np.float64),
        np.asarray(p2, dtype=np.float64),
    )

    scene.ngeom += 1


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

def plot_single_sensor_6axis(df, trial_idx, sensor_id):
    sub = df[
        (df["trial"] == trial_idx) &
        (df["sensor_id"] == sensor_id)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)

    signals = [
        ("Fx_N", "Fx [N]"),
        ("Fy_N", "Fy [N]"),
        ("Fz_N", "Fz [N]"),
        ("Tx_Nm", "Tx [N·m]"),
        ("Ty_Nm", "Ty [N·m]"),
        ("Tz_Nm", "Tz [N·m]"),
    ]

    for ax, (col, ylabel) in zip(axes.flatten(), signals):
        ax.plot(sub["time_s"], sub[col], linewidth=2)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    for ax in axes[1, :]:
        ax.set_xlabel("Time [s]")

    fig.suptitle(f"Trial {trial_idx:02d} | FT{sensor_id}", fontsize=14)
    plt.tight_layout()

    out_path = OUT_DIR / f"trial_{trial_idx:02d}_FT{sensor_id}_6axis.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_trial_all_sensors_fz(df, trial_idx):
    plt.figure(figsize=(9, 5))

    for sensor_id in range(1, 6):
        sub = df[
            (df["trial"] == trial_idx) &
            (df["sensor_id"] == sensor_id)
        ]

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


def save_and_plot_results(result_rows):
    if not result_rows:
        print("No FT data to save.")
        return

    df = pd.DataFrame(result_rows)
    df.to_csv(OUT_CSV, index=False)

    print("")
    print("Saved:", OUT_CSV.resolve())

    trials = sorted(df["trial"].unique())

    for trial_idx in trials:
        for sensor_id in range(1, 6):
            plot_single_sensor_6axis(df, trial_idx, sensor_id)

        plot_trial_all_sensors_fz(df, trial_idx)

    print(f"Saved plots to: {OUT_DIR.resolve()}")

# =====================================================
# Main
# =====================================================
def main():
    rows = read_csv_file(CSV_PATH)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print("Loaded model:", MODEL_PATH)
    print("Loaded rows:", len(rows))
    print("Controls:")
    print("  SPACE : start")
    print("  P     : pause/resume")
    print("  R     : reset")
    print("  ESC   : quit")

    state = {
        "started": False,
        "paused": False,
        "idx": 0,
        "last_switch": time.time(),
        "trial_pause_until": 0.0,
        "finished": False,
    }

    result_rows = []

    def key_callback(keycode):
        if keycode == ord(" "):
            if not state["finished"]:
                state["started"] = True
                state["paused"] = False
                state["last_switch"] = time.time()
                print("Replay started")

        elif keycode == ord("P"):
            state["paused"] = not state["paused"]
            state["last_switch"] = time.time()
            print("Paused" if state["paused"] else "Resumed")

        elif keycode == ord("R"):
            state["started"] = False
            state["paused"] = False
            state["idx"] = 0
            state["last_switch"] = time.time()
            state["trial_pause_until"] = 0.0
            state["finished"] = False
            result_rows.clear()
            print("Replay reset")

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:

        while viewer.is_running():
            row = rows[state["idx"]]

            trial_idx = int(row["trial"])
            sample_idx = int(row["sample"])

            q = np.array([
                f(row, "j1_rad"),
                f(row, "j2_rad"),
                f(row, "j3_rad"),
                f(row, "j4_rad"),
                f(row, "j5_rad"),
                f(row, "j6_rad"),
            ], dtype=float)

            set_robot_q(model, data, q)
            mujoco.mj_forward(model, data)

            contact = np.array([
                f(row, "x_m"),
                f(row, "y_m"),
                0.0,
            ], dtype=float)

            current_force_N = f(row, "current_force_N")
            force = np.array([0.0, 0.0, -current_force_N], dtype=float)

            ft_values, measured_force_sum_N, transmission_efficiency = compute_virtual_ft(
                contact=contact,
                sensors=SENSORS,
                force=force,
                trial_idx=trial_idx,
                sample_idx=sample_idx,
            )

            contact_world = plate_to_world_point(contact)
            force_world = plate_to_world_vec(force)
            force_end_world = contact_world + 0.015 * force_world

            viewer.user_scn.ngeom = 0

            # plate center
            add_sphere(
                viewer.user_scn,
                PLATE_WORLD_CENTER,
                0.010,
                [1.0, 0.0, 0.0, 1.0],
            )

            # current contact point
            add_sphere(
                viewer.user_scn,
                contact_world,
                0.012,
                [0.0, 1.0, 0.0, 1.0],
            )

            # force vector
            if current_force_N > 1e-6:
                add_capsule(
                    viewer.user_scn,
                    contact_world,
                    force_end_world,
                    0.003,
                    [1.0, 0.0, 0.0, 1.0],
                )

            # sensors
            for s in SENSORS:
                add_sphere(
                    viewer.user_scn,
                    plate_to_world_point(s),
                    0.012,
                    [0.0, 0.2, 1.0, 1.0],
                )

            viewer.sync()

            if not state["started"] or state["paused"] or state["finished"]:
                time.sleep(0.01)
                continue

            now = time.time()

            if now < state["trial_pause_until"]:
                time.sleep(0.01)
                continue

            if now - state["last_switch"] >= TARGET_HOLD_SEC:
                for sensor_idx in range(5):
                    Fx, Fy, Fz = ft_values[sensor_idx, 0:3]
                    Tx, Ty, Tz = ft_values[sensor_idx, 3:6]
                    weight = ft_values[sensor_idx, 6]

                    result_rows.append({
                        "trial": trial_idx,
                        "sample": sample_idx,
                        "time_s": f(row, "time_s"),
                        "target_name": row["target_name"],
                        "contact_x_m": contact[0],
                        "contact_y_m": contact[1],
                        "sensor_id": sensor_idx + 1,
                        "weight": weight,
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
                
                print(
                    f"Trial {trial_idx:02d} | "
                    f"sample {sample_idx:04d} | "
                    f"t={f(row, 'time_s'):.2f}s | "
                    f"F={current_force_N:.2f}N | "
                    f"measured={measured_force_sum_N:.2f}N | "
                    f"eff={transmission_efficiency*100:.1f}%"
                )

                next_idx = state["idx"] + 1

                if next_idx >= len(rows):
                    state["idx"] = len(rows) - 1
                    state["finished"] = True
                    state["paused"] = True
                    print("Replay finished.")
                    save_and_plot_results(result_rows)
                    continue

                current_trial = int(rows[state["idx"]]["trial"])
                next_trial = int(rows[next_idx]["trial"])

                state["idx"] = next_idx

                if next_trial != current_trial:
                    print(f"Trial {current_trial:02d} finished. Pause {TRIAL_PAUSE_SEC}s.")
                    state["trial_pause_until"] = time.time() + TRIAL_PAUSE_SEC

                state["last_switch"] = now


if __name__ == "__main__":
    main()