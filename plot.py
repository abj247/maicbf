import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PlotHelper:
    def __init__(self):
        pass

    @staticmethod
    def show_obstacles(obs, ax, z=[0, 6], alpha=0.6, color='deepskyblue'):
        for x1, y1, x2, y2 in obs:
            xs, ys = np.meshgrid([x1, x2], [y1, y2])
            zs = np.ones_like(xs)
            ax.plot_surface(xs, ys, zs * z[0], alpha=alpha, color=color)
            ax.plot_surface(xs, ys, zs * z[1], alpha=alpha, color=color)

            xs, zs = np.meshgrid([x1, x2], z)
            ys = np.ones_like(xs)
            ax.plot_surface(xs, ys * y1, zs, alpha=alpha, color=color)
            ax.plot_surface(xs, ys * y2, zs, alpha=alpha, color=color)

            ys, zs = np.meshgrid([y1, y2], z)
            xs = np.ones_like(ys)
            ax.plot_surface(xs * x1, ys, zs, alpha=alpha, color=color)
            ax.plot_surface(xs * x2, ys, zs, alpha=alpha, color=color)

    @staticmethod
    def plot_data(x, y, x_label, y_label, title, legend, file_name):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label=legend)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.savefig(file_name, dpi=300)
        plt.show()

    @staticmethod
    def save_to_csv(time_steps, values, csv_file_path):
        df = pd.DataFrame({
            'Time Steps': time_steps,
            'Values': values
        })
        df.to_csv(csv_file_path, index=False)
        print(f"CSV file has been saved to {csv_file_path}")