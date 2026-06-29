import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


RESULT_CSV = "./plate_virtual_ft_results_gaussian_transmission_130_150.csv"
SUMMARY_CSV = "./plate_virtual_ft_summary_gaussian_transmission_130_150.csv"

OUT_ERROR_BAR = "./plate_layout_error_bar_transmission_130_150.png"
OUT_ERROR_BOX = "./plate_layout_error_box_transmission_130_150.png"
OUT_FORCE_DIST = "./plate_force_distribution_transmission_130_150.png"
OUT_CONTACT_SCATTER = "./plate_contact_estimation_scatter_transmission_130_150.png"


def main():
    result = pd.read_csv(RESULT_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    # =====================================================
    # 1. Layout별 평균 / RMSE / 최대 오차 비교
    # =====================================================
    plt.figure(figsize=(10, 6))
    x = range(len(summary))

    plt.bar(x, summary["mean_error_mm"], width=0.25, label="Mean error")
    plt.bar([i + 0.25 for i in x], summary["rmse_error_mm"], width=0.25, label="RMSE")
    plt.bar([i + 0.50 for i in x], summary["max_error_mm"], width=0.25, label="Max error")

    plt.xticks([i + 0.25 for i in x], summary["layout"], rotation=30, ha="right")
    plt.ylabel("Error [mm]")
    plt.title("Contact estimation error by sensor layout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ERROR_BAR, dpi=200)
    print("Saved:", Path(OUT_ERROR_BAR).resolve())

    # =====================================================
    # 2. Layout별 error 분포 box plot
    # =====================================================
    error_data = []
    labels = []

    for layout, g in result.groupby("layout"):
        # sensor_id == 1만 쓰면 target당 error 하나만 남음
        errors = g[g["sensor_id"] == 1]["error_mm"].to_numpy()
        error_data.append(errors)
        labels.append(layout)

    plt.figure(figsize=(10, 6))
    plt.boxplot(error_data, labels=labels)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Error [mm]")
    plt.title("Error distribution by sensor layout")
    plt.tight_layout()
    plt.savefig(OUT_ERROR_BOX, dpi=200)
    print("Saved:", Path(OUT_ERROR_BOX).resolve())

    # =====================================================
    # 3. Layout별 평균 센서 힘 분포
    # =====================================================
    force_summary = (
        result
        .assign(force_mag=lambda d: (d["Fx"]**2 + d["Fy"]**2 + d["Fz"]**2) ** 0.5)
        .groupby(["layout", "sensor_id"])["force_mag"]
        .mean()
        .reset_index()
    )

    layouts = list(force_summary["layout"].unique())
    sensor_ids = [1, 2, 3, 4]

    plt.figure(figsize=(11, 6))
    width = 0.18
    x = range(len(layouts))

    for i, sid in enumerate(sensor_ids):
        values = []
        for layout in layouts:
            v = force_summary[
                (force_summary["layout"] == layout) &
                (force_summary["sensor_id"] == sid)
            ]["force_mag"].values
            values.append(v[0] if len(v) else 0)

        plt.bar(
            [j + (i - 1.5) * width for j in x],
            values,
            width,
            label=f"FT{sid}",
        )

    plt.xticks(x, layouts, rotation=30, ha="right")
    plt.ylabel("Mean force magnitude [N]")
    plt.title("Mean force distribution by sensor layout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FORCE_DIST, dpi=200)
    print("Saved:", Path(OUT_FORCE_DIST).resolve())

    # =====================================================
    # 4. 실제 접촉점 vs 추정 접촉점 산점도
    #    가장 error가 작은 layout 하나만 표시
    # =====================================================
    best_layout = summary.sort_values("mean_error_mm").iloc[0]["layout"]
    best = result[
        (result["layout"] == best_layout) &
        (result["sensor_id"] == 1)
    ]

    plt.figure(figsize=(7, 7))
    plt.scatter(best["contact_x_m"] * 1000, best["contact_y_m"] * 1000, label="Ground truth")
    plt.scatter(best["est_x_m"] * 1000, best["est_y_m"] * 1000, label="Estimated")

    for _, row in best.iterrows():
        plt.plot(
            [row["contact_x_m"] * 1000, row["est_x_m"] * 1000],
            [row["contact_y_m"] * 1000, row["est_y_m"] * 1000],
            linewidth=0.7,
        )

    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title(f"Contact estimation result: {best_layout}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CONTACT_SCATTER, dpi=200)
    print("Saved:", Path(OUT_CONTACT_SCATTER).resolve())

    print("")
    print("Best layout:")
    print(summary.sort_values("mean_error_mm").head(1))


if __name__ == "__main__":
    main()