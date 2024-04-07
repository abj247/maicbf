# MA-ICBF
### Installation
Create a virtual environment with Anaconda:
```bash
conda create -n maicbf python=3.10
```
Activate the virtual environment:
```bash
conda activate maicbf
```
Clone this repository:
```bash
git clone https://github.com/abj247/MA-ICBF.git
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

To bulk train the model with different number of agents, differemt loss weights for inpiut constraints and agility use this command,
```bash
python main.py --num_agents 4
```
this will train the ma-icbf model for ```4, 8, 16, 32``` agenst with ```0.5, 1.0, 1.5, 2.0```  loss weights for input constarints and ```0.1, 0.5, 1.0, 2.0``` loss weights for agility

Training logs can be found in [train_logs](https://github.com/abj247/MA-ICBF/tree/master/train_logs). For training and evaluating the baselines model see [baselines](https://github.com/abj247/MA-ICBF/tree/master/baselines).

### Evaluation
For evaluation of ma-icbf model use this command,
```bash
python evaluate.py --num_agents 16 --model_path models/agile_u_max_0.2/model_ours_weight_1.0_agents_4_v_max_0.2_u_max_0.2_sigma_0.05_default_iter_69999 --vis 1
```
This will evaluate the model for 16 agents with pretrained weights for 4 agents, save the cbf data to csv and show the visualization, for pretrained weights see [models](https://github.com/abj247/MA-ICBF/tree/master/models)

For evaluation with more adavanced capabilities uise this command,
```bash
python eval.py --num_agents 16 --model_path models/agile_u_max_0.2/model_ours_weight_1.0_agents_4_v_max_0.2_u_max_0.2_sigma_0.05_default_iter_69999 --vis 1
```
This will evaluate the pretrained model of 4 agents for 16 agents and will detect deadlock, track collision, save cbf data and time to goal for each agents in a csv file, for all csv data see [csv_data](https://github.com/abj247/MA-ICBF/tree/master/csv_data). The evaluation will give the h(u), velocity and accleration plots for each agent and all plots will be saved in [plots](https://github.com/abj247/MA-ICBF/tree/master/plots). 

To plot the trajectory for each agent for further visualization use this command,
```bash
python plot_traj.py
```
This script will plot the trajectory for each agent and it will required the ``pkl`` file for trajectory data which will be saved after evaluation completes see [trajectory](https://github.com/abj247/MA-ICBF/tree/master/trajectory). wandb logs could be found at [wandb](https://github.com/abj247/MA-ICBF/tree/master/wandb).

