import pickle
import matplotlib.pyplot as plt

# Replace 'your_file.pkl' with the path to your actual pickle file
file_path = 'trajectory/env_traj_eval.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Extracting the trajectory data for the first iteration
trajectory_data = data['trajectory'][0]  # Assuming the shape [x, 4, 5] where x is the number of steps

# Setting up the plot
plt.figure(figsize=(10, 8))

# Define colors for each agent for visibility
colors = ['red', 'green', 'blue', 'cyan']

for agent_index in range(4):  # Assuming 4 agents
    # Extracting the x and y coordinates for each agent
    xn = trajectory_data[:, agent_index, 0]  # x coordinates
    yn = trajectory_data[:, agent_index, 1]  # y coordinates

    # Plotting the trajectory for each agent
    plt.plot(xn, yn, color=colors[agent_index], marker='o', linestyle='-', label=f'Agent_{agent_index + 1}')

plt.title('Agent Trajectories for Evaluation')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend(loc='best')
plt.grid(False)
plt.savefig('plots/plot_traj.png', dpi=300)
plt.show()
