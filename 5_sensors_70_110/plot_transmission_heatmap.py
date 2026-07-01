import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH="./5_sensors_70_110/plate_virtual_ft_results_gaussian_transmission_70_110.csv"

df=pd.read_csv(CSV_PATH)

df=df[df.sensor_id==1]

layouts=df.layout.unique()

for layout in layouts:

    sub=df[df.layout==layout]

    fig=plt.figure(figsize=(6,5))

    plt.scatter(
        sub.contact_x_m*1000,
        sub.contact_y_m*1000,
        c=sub.transmission_efficiency*100,
        s=180,
        cmap="viridis",
        edgecolors="black"
    )

    plt.colorbar(label="Transmission efficiency (%)")

    plt.title(layout)

    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")

    plt.xlim(-150,150)
    plt.ylim(-150,150)

    plt.gca().set_aspect("equal")

    plt.grid(True)

    plt.savefig(
        f"plate_transmission_heatmap_{layout}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("Transmission heatmaps saved.")