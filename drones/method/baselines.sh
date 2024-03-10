#!/bin/bash

# run each baseline methood

run_pic() {
    read -p "Enter experiment name (e.g., coop_navigation_n6): " exp_name
    read -p "Enter scenario (e.g., simple_spread_n6): " scenario
    read -p "Enter critic type (e.g., gcn_max): " critic_type
    echo "Running PIC with experiment name $exp_name, scenario $scenario, and critic type $critic_type..."
    python PIC/maddpg/main_vec.py --exp_name "$exp_name" --scenario "$scenario" --critic_type "$critic_type" --cuda                             
    #python PIC/maddpg/main_vec.py --exp_name coop_navigation_n6 --scenario simple_spread_n6  --critic_type gcn_max  --cuda
    echo "PIC execution completed."
}

run_gcbf_pytorch() {
    read -p "Enter algorithm (e.g., gcbf): " algo
    read -p "Enter environment (e.g., DubinsCar): " env
    read -p "Enter number of agents (-n, e.g., 16): " n
    read -p "Enter steps (e.g., 500000): " steps
    echo "Running gcbf-pytorch with algo $algo, environment $env, agents $n, steps $steps..."
    python gcbf-pytorch/train.py --algo "$algo" --env "$env" -n "$n" --steps "$steps"
    #python gcbf-pytorch/train.py --algo gcbf --env DubinsCar -n 16 --steps 500000
    echo "gcbf-pytorch execution completed."
}

run_off_policy() {
    echo "Configuring off-policy method..."
    read -p "Enter environment name (e.g., MPE): " env_name
    read -p "Enter algorithm name (e.g., rmappo): " algorithm_name
    read -p "Enter experiment name: " experiment_name
    read -p "Enter scenario name: " scenario_name
    read -p "Enter number of agents: " num_agents
    read -p "Enter number of landmarks: " num_landmarks
    read -p "Enter seed: " seed
    read -p "Enter number of rollout threads: " n_rollout_threads
    read -p "Enter episode length: " episode_length
    read -p "Enter actor train interval step: " actor_train_interval_step
    read -p "Enter tau: " tau
    read -p "Enter learning rate (lr): " lr
    read -p "Enter number of environment steps: " num_env_steps
    read -p "Enter batch size: " batch_size
    read -p "Enter buffer size: " buffer_size
    read -p "Use reward normalization? (yes/no): " use_reward_normalization
    read -p "Use wandb? (yes/no): " use_wandb

    
    if [[ $use_reward_normalization == "yes" ]]; then
        use_reward_normalization="--use_reward_normalization"
    else
        use_reward_normalization=""
    fi
    if [[ $use_wandb == "yes" ]]; then
        use_wandb="--use_wandb"
    else
        use_wandb=""
    fi

    echo "Running off-policy method..."
    python off-policy/train/train_mpe.py --env_name "$env_name" --algorithm_name "$algorithm_name" --experiment_name "$experiment_name" --scenario_name "$scenario_name" --num_agents "$num_agents" --num_landmarks "$num_landmarks" --seed "$seed" --n_rollout_threads "$n_rollout_threads" --episode_length "$episode_length" --actor_train_interval_step "$actor_train_interval_step" --tau "$tau" --lr "$lr" --num_env_steps "$num_env_steps" --batch_size "$batch_size" --buffer_size "$buffer_size" $use_reward_normalization $use_wandb
    #python off-policy/train/train_mpe.py --env_name MPE --algorithm_name rmappo --experiment_name debug --scenario_name simple_spread --num_agents 2 --num_landmarks 2 --seed 1 --n_rollout_threads 128 --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 10000000 --batch_size 1000 --buffer_size 500000 --use_reward_normalization --use_wandb
    #     
}


run_on_policy() {
    echo "Configuring on-policy method..."
    read -p "Enter environment name (e.g., MPE): " env_name
    read -p "Enter algorithm name (e.g., rmappo): " algorithm_name
    read -p "Enter experiment name: " experiment_name
    read -p "Enter scenario name: " scenario_name
    read -p "Enter number of agents: " num_agents
    read -p "Enter number of landmarks: " num_landmarks
    read -p "Enter seed: " seed
    read -p "Enter number of training threads: " n_training_threads
    read -p "Enter number of rollout threads: " n_rollout_threads
    read -p "Enter number of mini batches: " num_mini_batch
    read -p "Enter episode length: " episode_length
    read -p "Enter number of environment steps: " num_env_steps
    read -p "Enter PPO epoch: " ppo_epoch
    read -p "Use ReLU? (yes/no): " use_ReLU
    read -p "Enter gain: " gain
    read -p "Enter learning rate (lr): " lr
    read -p "Enter critic learning rate: " critic_lr
    read -p "Enter wandb name: " wandb_name
    read -p "Enter user name: " user_name

    # Convert yes/no answers to boolean flags
    if [[ $use_ReLU == "yes" ]]; then
        use_ReLU="--use_ReLU"
    else
        use_ReLU=""
    fi

    echo "Running on-policy method..."
    python on-policy/onpolicy/scripts/train/train_mpe.py --env_name "$env_name" --algorithm_name "$algorithm_name" --experiment_name "$experiment_name" --scenario_name "$scenario_name" --num_agents "$num_agents" --num_landmarks "$num_landmarks" --seed "$seed" --n_training_threads "$n_training_threads" --n_rollout_threads "$n_rollout_threads" --num_mini_batch "$num_mini_batch" --episode_length "$episode_length" --num_env_steps "$num_env_steps" --ppo_epoch "$ppo_epoch" $use_ReLU --gain "$gain" --lr "$lr" --critic_lr "$critic_lr" --wandb_name "$wandb_name" --user_name "$user_name"
    #python on-policy/onpolicy/scripts/train/train_mpe.py --env_name MPE --algorithm_name rmappo --experiment_name check --scenario_name simple_spread --num_agents 2 --num_landmarks 3 --seed 1  --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000  --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "xxx" --user_name "yyy"
    
}

run_infomarl() {
    echo "Configuring InfoMARL method..."
    read -p "Use value normalization? (yes/no): " use_valuenorm
    read -p "Use POPART? (yes/no): " use_popart
    read -p "Enter project name: " project_name
    read -p "Enter environment name: " env_name
    read -p "Enter algorithm name: " algorithm_name
    read -p "Enter seed: " seed
    read -p "Enter experiment name: " experiment_name
    read -p "Enter scenario name: " scenario_name
    read -p "Enter number of agents: " num_agents
    read -p "Enter collision reward: " collision_rew
    read -p "Enter number of training threads: " n_training_threads
    read -p "Enter number of rollout threads: " n_rollout_threads
    read -p "Enter number of mini batches: " num_mini_batch
    read -p "Enter episode length: " episode_length
    read -p "Enter number of environment steps: " num_env_steps
    read -p "Enter PPO epoch: " ppo_epoch
    read -p "Use ReLU? (yes/no): " use_ReLU
    read -p "Enter gain: " gain
    read -p "Enter learning rate: " lr
    read -p "Enter critic learning rate: " critic_lr
    read -p "Enter user name: " user_name
    read -p "Use central observation? (true/false): " use_cent_obs
    read -p "Enter graph feature type: " graph_feat_type
    read -p "Auto mini batch size? (yes/no): " auto_mini_batch_size
    read -p "Enter target mini batch size: " target_mini_batch_size

    # Convert yes/no answers to boolean flags or appropriate command line arguments
    use_valuenorm=$( [[ $use_valuenorm == "yes" ]] && echo "--use_valuenorm" || echo "" )
    use_popart=$( [[ $use_popart == "yes" ]] && echo "--use_popart" || echo "" )
    use_ReLU=$( [[ $use_ReLU == "yes" ]] && echo "--use_ReLU" || echo "" )
    auto_mini_batch_size=$( [[ $auto_mini_batch_size == "yes" ]] && echo "--auto_mini_batch_size" || echo "" )

    echo "Running InfoMARL method..."
    python -u InfoMARL/onpolicy/scripts/train_mpe.py $use_valuenorm $use_popart --project_name "$project_name" --env_name "$env_name" --algorithm_name "$algorithm_name" --seed "$seed" --experiment_name "$experiment_name" --scenario_name "$scenario_name" --num_agents "$num_agents" --collision_rew "$collision_rew" --n_training_threads "$n_training_threads" --n_rollout_threads "$n_rollout_threads" --num_mini_batch "$num_mini_batch" --episode_length "$episode_length" --num_env_steps "$num_env_steps" --ppo_epoch "$ppo_epoch" $use_ReLU --gain "$gain" --lr "$lr" --critic_lr "$critic_lr" --user_name "$user_name" --use_cent_obs "$use_cent_obs" --graph_feat_type "$graph_feat_type" $auto_mini_batch_size --target_mini_batch_size "$target_mini_batch_size"
    #python -u InfoMARL/onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart --project_name "informarl" --env_name "GraphMPE" --algorithm_name "rmappo" --seed 0 --experiment_name "informarl" --scenario_name "navigation_graph" --num_agents 3 --collision_rew 5 --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 2000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --user_name "marl" --use_cent_obs "False" --graph_feat_type "relative" --auto_mini_batch_size --target_mini_batch_size 128
}

echo "Select the method you want to run:"
echo "1. PIC"
echo "2. gcbf-pytorch"
echo "3. off-policy"
echo "4. on-policy"
echo "5. InfoMARL"
read -p "Enter your choice (1-5): " choice

case $choice in
    1) run_pic ;;
    2) run_gcbf_pytorch ;;
    3) run_off_policy ;;
    4) run_on_policy ;;
    5) run_infomarl ;;
    *) echo "Invalid choice. Please run the script again and select a valid option." ;;
esac





