<div align="center">

# MA-ICBF

*Decentralized Safe and Scalable Multi-Agent Control Under Limited Actuation*

</div>

## Abstract:

In this paper, we consider the problem of safe and deadlock-free control for multiple agents operating in an environment with obstacles and other agents under limited actuation. We propose a joint-learning framework where the decentralized control policy and Integral Control Barrier Functions (ICBF), which act as safety certificates for input-constrained nonlinear systems, are learned simultaneously. As the neural control policy provides only probabilistic guarantees for collision avoidance, we embed a novel computationally light decentralized MPC Integral Control Barrier Functions (MPC-ICBF) into the neural network policy to guarantee safety as well as scalability. By drawing an analogy between minimizing non-convex loss functions by well-established gradient methods in machine learning and the problem of local minima, we propose a novel approach that is augmented with the proposed learning framework to minimize deadlocks. Numerical simulations demonstrate that our approach outperforms other state of the art multi-agent control algorithms in terms of safety, input constraint satisfaction, and number of deadlocks. Furthermore, we show good generalization capabilities across different scenarios with varying numbers of agents up to 1000 agents.

### Installation
Create a virtual environment with Anaconda:
```bash
conda create -n maicbf python=3.6
```
Activate the virtual environment:
```bash
sourve activate maicbf
```
Clone this repository:
```bash
git clone https://github.com/abj247/maicbf.git
```
Enter the main folder and install the dependencies:
```bash
pip install -r requirements.txt
```
### Training
To train the ma-icbf model specified number of agents (e.g. 4) use this command,
```bash
python train.py --num_agents 4
```
This will train the model for 4 agents 

To bulk train the model with different number of agents, different loss weights for inpiut constraints and agility use this command,
```bash
python main.py --num_agents 4
```
this will train the ma-icbf model for ```4, 8, 16, 32``` agenst with ```0.5, 1.0, 1.5, 2.0```  loss weights for input constraints and ```0.1, 0.5, 1.0, 2.0``` loss weights for agility.

Training logs can be found in [train_logs](https://github.com/abj247/MA-ICBF/tree/master/train_logs). For training and evaluating the baselines models see [baselines](https://github.com/abj247/MA-ICBF/tree/master/baselines).

### Evaluation
For evaluation of ma-icbf model use this command,
```bash
python evaluate.py --num_agents 16 --model_path models/agile_u_max_0.2/model_ours_weight_1.0_agents_4_v_max_0.2_u_max_0.2_sigma_0.05_default_iter_69999 --vis 1
```
This will evaluate the model for 16 agents with pretrained weights for 4 agents, save the cbf data to csv and show the visualization, for pretrained weights see [models](https://github.com/abj247/MA-ICBF/tree/master/models).

For evaluation with more advanced capabilities use this command,
```bash
python eval.py --num_agents 16 --model_path models/agile_u_max_0.2/model_ours_weight_1.0_agents_4_v_max_0.2_u_max_0.2_sigma_0.05_default_iter_69999 --vis 1
```
This will evaluate the pretrained model of 4 agents for 16 agents and will detect deadlock, resolve deadlock track collision, resolve collision for all agents using mpc-cbf controller save cbf data and time to goal for each agents in a csv file, for all csv data see [csv_data](https://github.com/abj247/MA-ICBF/tree/master/csv_data). The evaluation will give the h(u), velocity and acceleration plots for each agent and all plots will be saved in [plots](https://github.com/abj247/MA-ICBF/tree/master/plots). 

To plot the trajectory for each agent for further visualization use this command,
```bash
python plot_traj.py
```
This script will plot the trajectory for each agent and it will be required to input the ``.pkl`` file for trajectory data which will be saved after evaluation completes see [trajectory](https://github.com/abj247/MA-ICBF/tree/master/trajectory). wandb logs could be found at [wandb](https://github.com/abj247/MA-ICBF/tree/master/wandb).

