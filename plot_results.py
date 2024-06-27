import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(results_file):
    # Read the results file
    df = pd.read_csv(results_file)
    
    # Convert columns to appropriate data types
    df['num_agents'] = df['num_agents'].astype(int)
    df['deadlocked_agents'] = df['deadlocked_agents'].astype(float)
    df['mean_safety_ratio'] = df['mean_safety_ratio'].astype(float)
    df['time'] = df['time'].astype(float)
    df['num_outside_constraints'] = df['num_outside_constraints'].astype(int)
    
    # Define metrics to plot
    metrics = ['deadlocked_agents', 'mean_safety_ratio', 'time', 'num_outside_constraints']
    titles = ['Number of Deadlocked Agents', 'Mean Safety Ratio (Learning)', 'Time', 'Number of Agents Outside Input Constraint']
    
    # Create plots
    for metric, title in zip(metrics, titles):
        plt.figure()
        for method in df['method'].unique():
            subset = df[df['method'] == method]
            plt.plot(subset['num_agents'], subset[metric], label=method)
        plt.xlabel('Number of Agents')
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{metric}.png')
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_results.py <results_file>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    plot_metrics(results_file)
