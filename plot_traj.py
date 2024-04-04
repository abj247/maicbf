
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('csv_data/trajectory/trajectory.csv')

# Determine the number of agents by dividing the number of columns by 2
num_agents = df.shape[1] // 2

# Plotting
plt.figure(figsize=(10, 6))

for i in range(num_agents):
    # Extract x and y coordinates for each agent
    x_column = f'Agent_{i+1}_x'
    y_column = f'Agent_{i+1}_y'
    plt.plot(df[x_column], df[y_column], label=f'Agent_{i+1}')

plt.title('Agents Trajectories for all agents')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()
