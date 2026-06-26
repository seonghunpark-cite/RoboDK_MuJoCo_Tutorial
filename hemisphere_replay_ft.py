import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = "./assets/fairino_description/urdf/fairino5_v6_converted.xml"
CSV_PATH = "./robodk_press_targets.csv"

HEMI_WORLD_CENTER = np.array([0.45, 0.0, 0.25])
HEMI_TO_WORLD_R = np.eye(3)

INNER_RADIUS = 0.20
SENSOR_RADIUS = 0.21

TARGET_FORCE_N = 10.0
FORCE_SIGN = 1.0
ALPHA = 2.0

TARGET_HOLD_SEC = 1.0
EE_BODY_NAME = "wrist3_link"

RESULT_CSV = "./mujoco_virtual_ft_results.csv"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = [
        "target_name",
        "j1_rad", "j2_rad", "j3_rad", "j4_rad", "j5_rad", "j6_rad",
        "x_m", "y_m", "z_m",
        "nx", "ny", "nz",
    ]

    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    return rows


def f(row, key):
    return float(row[key])


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v), 0.0
    return v / n, n


def hemi_to_world_point(p_hemi):
    return HEMI_WORLD_CENTER + HEMI_TO_WORLD_R @ p_hemi


def hemi_to_world_vec(v_hemi):
    return HEMI_TO_WORLD_R @ v_hemi


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


class VirtualFTHemisphere:
    def __init__(self, sensor_radius=0.21, alpha=2.0):
        self.sensor_radius = sensor_radius
        self.alpha = alpha

        s = np.sqrt(0.5)

        self.sensor_dirs = np.array([
            [0.0, 0.0, -1.0],
            [s,   0.0, -s],
            [0.0, s,   -s],
            [-s,  0.0, -s],
            [0.0, -s,  -s],
        ], dtype=float)

        self.sensor_pos = self.sensor_radius * self.sensor_dirs

    def compute(self, contact_point, force):
        n, _ = normalize(contact_point)

        dots = self.sensor_dirs @ n
        weights = np.maximum(0.0, dots) ** self.alpha

        if np.sum(weights) < 1e-12:
            distances = np.linalg.norm(self.sensor_dirs - n, axis=1)
            weights = 1.0 / np.maximum(distances, 1e-6)

        weights = weights / np.sum(weights)

        values = []

        for i in range(5):
            Fi = weights[i] * force
            ri = contact_point - self.sensor_pos[i]
            Ti = np.cross(ri, Fi)
            values.append([Fi[0], Fi[1], Fi[2], Ti[0], Ti[1], Ti[2]])

        return np.array(values), weights


def main():
    rows = read_csv(CSV_PATH)

    ft_model = VirtualFTHemisphere(
        sensor_radius=SENSOR_RADIUS,
        alpha=ALPHA,
    )

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Loaded targets: {len(rows)}")
    print("Controls:")
    print("  SPACE : start")
    print("  P     : pause/resume")
    print("  R     : reset")

    result_rows = []

    state = {
        "started": False,
        "paused": False,
        "target_idx": 0,
        "last_switch": time.time(),
        "printed_current": False,
    }

    def key_callback(keycode):
        if keycode == ord(" "):
            state["started"] = True
            state["paused"] = False
            state["last_switch"] = time.time()
            state["printed_current"] = False
            print("Replay started")

        elif keycode == ord("P"):
            state["paused"] = not state["paused"]
            state["last_switch"] = time.time()
            print("Paused" if state["paused"] else "Resumed")

        elif keycode == ord("R"):
            state["started"] = False
            state["paused"] = False
            state["target_idx"] = 0
            state["last_switch"] = time.time()
            state["printed_current"] = False
            print("Replay reset")

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:

        while viewer.is_running():
            row = rows[state["target_idx"]]

            q = np.array([
                f(row, "j1_rad"),
                f(row, "j2_rad"),
                f(row, "j3_rad"),
                f(row, "j4_rad"),
                f(row, "j5_rad"),
                f(row, "j6_rad"),
            ])

            set_robot_q(model, data, q)
            mujoco.mj_forward(model, data)

            tcp_hemi = np.array([
                f(row, "x_m"),
                f(row, "y_m"),
                f(row, "z_m"),
            ])

            tcp_dir, _ = normalize(tcp_hemi)
            contact_hemi = INNER_RADIUS * tcp_dir

            normal_hemi = np.array([
                f(row, "nx"),
                f(row, "ny"),
                f(row, "nz"),
            ])
            normal_hemi, _ = normalize(normal_hemi)

            force_hemi = FORCE_SIGN * TARGET_FORCE_N * normal_hemi

            sensor_values, weights = ft_model.compute(
                contact_point=contact_hemi,
                force=force_hemi,
            )

            contact_world = hemi_to_world_point(contact_hemi)
            tcp_world = hemi_to_world_point(tcp_hemi)
            force_world = hemi_to_world_vec(force_hemi)
            force_end_world = contact_world + 0.008 * force_world

            viewer.user_scn.ngeom = 0

            add_sphere(
                viewer.user_scn,
                HEMI_WORLD_CENTER,
                0.010,
                [1.0, 0.0, 0.0, 1.0],
            )

            add_sphere(
                viewer.user_scn,
                tcp_world,
                0.010,
                [1.0, 1.0, 0.0, 1.0],
            )

            add_sphere(
                viewer.user_scn,
                contact_world,
                0.012,
                [0.0, 1.0, 0.0, 1.0],
            )

            add_capsule(
                viewer.user_scn,
                contact_world,
                force_end_world,
                0.003,
                [1.0, 0.0, 0.0, 1.0],
            )

            for sdir in ft_model.sensor_dirs:
                sensor_world = hemi_to_world_point(SENSOR_RADIUS * sdir)
                add_sphere(
                    viewer.user_scn,
                    sensor_world,
                    0.010,
                    [0.0, 0.2, 1.0, 1.0],
                )

            viewer.sync()

            if not state["started"] or state["paused"]:
                time.sleep(0.01)
                continue

            now = time.time()
            if now - state["last_switch"] >= TARGET_HOLD_SEC:
                print("")
                print("=" * 70)
                print(f"{state['target_idx'] + 1}/{len(rows)} {row['target_name']}")
                print("tcp_hemi      =", np.round(tcp_hemi, 4))
                print("contact_hemi  =", np.round(contact_hemi, 4))
                print("normal_hemi   =", np.round(normal_hemi, 4))
                print("force_hemi[N] =", np.round(force_hemi, 4))
                print("-" * 70)

                for i, s in enumerate(sensor_values):
                    print(
                        f"FT{i+1} "
                        f"w={weights[i]:.3f} | "
                        f"Fx={s[0]:8.3f} "
                        f"Fy={s[1]:8.3f} "
                        f"Fz={s[2]:8.3f} | "
                        f"Tx={s[3]:8.4f} "
                        f"Ty={s[4]:8.4f} "
                        f"Tz={s[5]:8.4f}"
                    )

                    result_rows.append([
                        row["target_name"],
                        i + 1,
                        weights[i],
                        s[0], s[1], s[2],
                        s[3], s[4], s[5],
                        contact_hemi[0],
                        contact_hemi[1],
                        contact_hemi[2],
                        normal_hemi[0],
                        normal_hemi[1],
                        normal_hemi[2],
                    ])

                state["target_idx"] += 1

                if state["target_idx"] >= len(rows):
                    state["target_idx"] = len(rows) - 1
                    state["paused"] = True
                    print("Replay finished")

                state["last_switch"] = now

    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "target_name",
            "sensor_id",
            "weight",
            "Fx", "Fy", "Fz",
            "Tx", "Ty", "Tz",
            "contact_x_m", "contact_y_m", "contact_z_m",
            "nx", "ny", "nz",
        ])
        writer.writerows(result_rows)

    print(f"Saved: {RESULT_CSV}")

if __name__ == "__main__":
    main()