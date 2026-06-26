import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RESULT_CSV = "./mujoco_virtual_ft_results.csv"
OUT_SUMMARY_CSV = "./ft_target_summary.csv"
OUT_FORCE_PLOT = "./ft_force_distribution.png"
OUT_ERROR_PLOT = "./ft_contact_error.png"

SENSOR_RADIUS = 0.21

SENSOR_DIRS = np.array([
    [0.0, 0.0, -1.0],
    [np.sqrt(0.5), 0.0, -np.sqrt(0.5)],
    [0.0, np.sqrt(0.5), -np.sqrt(0.5)],
    [-np.sqrt(0.5), 0.0, -np.sqrt(0.5)],
    [0.0, -np.sqrt(0.5), -np.sqrt(0.5)],
], dtype=float)

SENSOR_POS = SENSOR_RADIUS * SENSOR_DIRS


def estimate_contact_from_sensor_forces(group):
    forces = group[["Fx", "Fy", "Fz"]].to_numpy()
    mags = np.linalg.norm(forces, axis=1)

    if np.sum(mags) < 1e-12:
        return np.zeros(3), np.nan

    estimated = np.sum(SENSOR_POS * mags[:, None], axis=0) / np.sum(mags)
    return estimated, np.linalg.norm(estimated)


def main():
    df = pd.read_csv(RESULT_CSV)

    summaries = []

    for target, g in df.groupby("target_name", sort=False):
        g = g.sort_values("sensor_id")

        forces = g[["Fx", "Fy", "Fz"]].to_numpy()
        force_mags = np.linalg.norm(forces, axis=1)
        force_sum = forces.sum(axis=0)
        force_sum_mag = np.linalg.norm(force_sum)

        contact = g[["contact_x_m", "contact_y_m", "contact_z_m"]].iloc[0].to_numpy()
        normal = g[["nx", "ny", "nz"]].iloc[0].to_numpy()

        est_contact, est_radius = estimate_contact_from_sensor_forces(g)
        error = np.linalg.norm(est_contact - contact)

        row = {
            "target_name": target,
            "force_sum_N": force_sum_mag,
            "contact_x_m": contact[0],
            "contact_y_m": contact[1],
            "contact_z_m": contact[2],
            "est_x_m": est_contact[0],
            "est_y_m": est_contact[1],
            "est_z_m": est_contact[2],
            "contact_error_m": error,
            "contact_error_mm": error * 1000.0,
            "nx": normal[0],
            "ny": normal[1],
            "nz": normal[2],
        }

        for i, mag in enumerate(force_mags, start=1):
            row[f"FT{i}_force_mag_N"] = mag

        summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    print("Saved:", Path(OUT_SUMMARY_CSV).resolve())
    print("")
    print(summary[[
        "target_name",
        "force_sum_N",
        "contact_error_mm",
        "FT1_force_mag_N",
        "FT2_force_mag_N",
        "FT3_force_mag_N",
        "FT4_force_mag_N",
        "FT5_force_mag_N",
    ]])

    # Plot 1: force distribution by target
    x = np.arange(len(summary))
    width = 0.15

    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.bar(
            x + (i - 2) * width,
            summary[f"FT{i+1}_force_mag_N"],
            width,
            label=f"FT{i+1}",
        )

    plt.xticks(x, summary["target_name"], rotation=45, ha="right")
    plt.ylabel("Force magnitude [N]")
    plt.title("Virtual FT force distribution by target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FORCE_PLOT, dpi=200)
    print("Saved:", Path(OUT_FORCE_PLOT).resolve())

    # Plot 2: contact estimation error
    plt.figure(figsize=(10, 5))
    plt.bar(summary["target_name"], summary["contact_error_mm"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Contact estimation error [mm]")
    plt.title("Estimated contact point error from virtual FT distribution")
    plt.tight_layout()
    plt.savefig(OUT_ERROR_PLOT, dpi=200)
    print("Saved:", Path(OUT_ERROR_PLOT).resolve())


if __name__ == "__main__":
    main()
