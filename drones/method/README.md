# Baseline Methods for Multi-Agent Reinforcement Learning

This repository contains a collection of baseline methods for multi-agent reinforcement learning (MARL) experiments. It includes scripts to run different algorithms like PIC, gcbf-pytorch, off-policy, on-policy, and InfoMARL across various scenarios.

## Environment Setup

1. **Clone the Repository**: First, clone this repository to your local machine and navigate to baselines folder.
    ```
    git clone https://github.com/abj247/MA-ICBF.git
    cd MA-ICBF/baselines
    ```
2. **Create a Conda Environment**: It is recommended to create a virtual environment for managing the dependencies.
    ```
    conda create -n baselines python=3.6
    ```
3. **Install Dependencies**: Install the required Python packages using the `requirements.txt` file.
    ```
    pip install -r requirements.txt
    ```

## Running the Baseline Methods

To run all the baseline methods, execute the provided shell script. This scripts interactively asks for parameters required by each method and runs them sequentially. 

### Running All Methods Sequentially

1. Make sure the script is executable. If not, you can make it executable by running:
    ```
    chmod +x baselines_run_all.sh
    ```
2. Run the script by executing:
    ```
    ./baselines_run_all.sh
    ```

### Running Individual Methods

You can also run individual methods by invoking their respective functions within the script. Below are the commands for each method:

1. Make sure the script is executable. If not, you can make it executable by running:
    ```
    chmod +x baselines.sh
    ```
2. Run the script by executing:
    ```
    ./baselines.sh

3. When prompted, enter the number corresponding to the method you wish to run. The options are:
    - `1` for PIC
    - `2` for gcbf-pytorch
    - `3` for off-policy
    - `4` for on-policy
    - `5` for InfoMARL

Follow the interactive prompts to input the required parameters for the selected method.

4. Example commands for running individual methods

- **PIC**: 
    ```
    python PIC/maddpg/main_vec.py --exp_name coop_navigation_n6 --scenario simple_spread_n6  --critic_type gcn_max  --cuda
    
    ```
- **gcbf-pytorch**: 
    ```
    python gcbf-pytorch/train.py --algo gcbf --env DubinsCar -n 16 --steps 500000
    
    ```
- **Off-Policy Method**: 
    ```
    python off-policy/train/train_mpe.py --env_name MPE --algorithm_name rmappo \
    --experiment_name debug --scenario_name simple_spread \
    --num_agents 2 --num_landmarks 2 --seed 1 --n_rollout_threads 128\
     --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 \
    --num_env_steps 10000000 --batch_size 1000 --buffer_size 500000 --use_reward_normalization --use_wandb
    
    ```
- **On-Policy Method**: 
    ```
    python on-policy/onpolicy/scripts/train/train_mpe.py --env_name MPE --algorithm_name rmappo\
     --experiment_name check --scenario_name simple_spread \
     --num_agents 2 --num_landmarks 3 --seed 1  --n_training_threads 1 \
    --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
     --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 \
    --critic_lr 7e-4 --wandb_name "xxx" --user_name "yyy"
    
    ```
- **InfoMARL**: 
    ```
    python -u InfoMARL/onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
     --project_name "informarl" --env_name "GraphMPE" \
     --algorithm_name "rmappo" --seed 0 --experiment_name "informarl" \
    --scenario_name "navigation_graph" --num_agents 3 --collision_rew 5 \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 \
    --episode_length 25 --num_env_steps 2000000 --ppo_epoch 10 --use_ReLU \
    --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "marl" --use_cent_obs "False" \
    --graph_feat_type "relative" --auto_mini_batch_size --target_mini_batch_size 128
    
    ```



