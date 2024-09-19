# Baseline Methods for Multi-Agent Navigation

This repository contains a collection of baseline methods for multi-agent reinforcement learning (MARL) experiments. It includes scripts to run different algorithms like PIC, gcbf-pytorch, macbf, and InfoMARL across various scenarios.

## Environment Setup

1. **Clone the Repository**: First, clone this repository to your local machine and navigate to baselines folder.
    ```
    git clone https://github.com/abj247/maicbf.git
    cd maicbf/baselines
    ```


4. Example commands for running each  individual baseline methods

- **PIC**: 
    ```
    python PIC/maddpg/main_vec.py \
    --exp_name coop_navigation_n6 \
    --scenario simple_spread_n6  \
    --critic_type gcn_max \
     --cuda
    
    ```
- **gcbf-pytorch**: 
    ```
    python gcbf-pytorch/train.py \
    --algo gcbf \
    --env DubinsCar \
    -n 16 \
    --steps 500000
    
    ```
  
- **MA-CBF**: 
    ```
    python evaluate.py --num_agents 32 --model_path models/model_save --vis 1
    
    ```
- **InforMARL**: 
    ```
    python -u InforMARL/onpolicy/scripts/train_mpe.py \
     --use_valuenorm \
     --use_popart \
     --project_name "informarl" \
     --env_name "GraphMPE" \
     --algorithm_name "rmappo" \
     --seed 0 \
     --experiment_name "informarl" \
    --scenario_name "navigation_graph" \
     --num_agents 3 \
     --collision_rew 5 \
    --n_training_threads 1 \
     --n_rollout_threads 128 \
     --num_mini_batch 1 \
    --episode_length 25 \
     --num_env_steps 2000000 \
    --ppo_epoch 10 \
     --use_ReLU \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --user_name "marl" \
    --use_cent_obs "False" \
    --graph_feat_type "relative" \
    --auto_mini_batch_size \
    --target_mini_batch_size 128
    
    ```



