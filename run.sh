#!/bin/bash

# Define variables
NUM_AGENTS=8
MODEL_PATH=models/agile_u_max_0.2/model_ours_weight_1.0_agents_4_v_max_0.2_u_max_0.2_sigma_0.05_default_iter_69999
GCBF_PATH=./baselines/gcbf-pytorch
GCBF_WEIGHTS_PATH=<path_to_gcbf_weights>  # Replace with the actual path to the weights
NUM_EPISODES=100  # Example value, replace with the actual number of episodes

# Run the first script (maicbf)
echo "Running MAICBF evaluation..."
MAICBF_OUTPUT=$(python new.py --num agents $NUM_AGENTS --model_path $MODEL_PATH)

# Extract metrics from the first script output
MAICBF_DEADLOCKED_AGENTS=$(echo "$MAICBF_OUTPUT" | grep -oP 'Deadlocked Agents: \K[\d.]+')
MAICBF_MEAN_SAFETY_RATIO=$(echo "$MAICBF_OUTPUT" | grep -oP 'Mean Safety Ratio \(Learning \| Baseline\): \K[\d.]+(?= \|)')
MAICBF_TIME=$(echo "$MAICBF_OUTPUT" | grep -oP 'Time: \K[\d.]+')

# Extract control matrix values and count values greater than 0.21
CONTROL_MATRIX_VALUES=$(echo "$MAICBF_OUTPUT" | grep -oP '\[.*\]')
NUM_OUTSIDE_CONSTRAINTS=$(echo "$CONTROL_MATRIX_VALUES" | grep -oP '[\d.]+' | awk '{if ($1 > 0.21) count++} END {print count}')

# Run the second script (macbf)
echo "Running MACBF evaluation..."
MACBF_OUTPUT=$(python new.py --num agents $NUM_AGENTS --model_path $MODEL_PATH)

# Extract metrics from the second script output
MACBF_DEADLOCKED_AGENTS=$(echo "$MACBF_OUTPUT" | grep -oP 'Deadlocked Agents: \K[\d.]+')
MACBF_MEAN_SAFETY_RATIO=$(echo "$MACBF_OUTPUT" | grep -oP 'Mean Safety Ratio \(Learning \| Baseline\): \K[\d.]+(?= \|)')
MACBF_TIME=$(echo "$MACBF_OUTPUT" | grep -oP 'Time: \K[\d.]+')

# Extract control matrix values and count values greater than 0.21
CONTROL_MATRIX_VALUES=$(echo "$MACBF_OUTPUT" | grep -oP '\[.*\]')
NUM_OUTSIDE_CONSTRAINTS_MACBF=$(echo "$CONTROL_MATRIX_VALUES" | grep -oP '[\d.]+' | awk '{if ($1 > 0.21) count++} END {print count}')

# Run the third script (gcbf)
echo "Running GCBF evaluation..."
GCBF_OUTPUT=$(python $GCBF_PATH/test.py --path $GCBF_WEIGHTS_PATH --epi $NUM_EPISODES -n $NUM_AGENTS --agility)

# Extract metrics from the third script output
GCBF_DEADLOCK_AVOIDANCE=$(echo "$GCBF_OUTPUT" | grep -oP 'reach rate: \K[\d.]+')
GCBF_MEAN_SAFETY_RATIO=$(echo "$GCBF_OUTPUT" | grep -oP 'safe rate: \K[\d.]+')
GCBF_TIME=$(echo "$GCBF_OUTPUT" | grep -oP 'Done in \K[\d]+')

# Output the metrics
echo "MAICBF Results:"
echo "Deadlocked Agents: $MAICBF_DEADLOCKED_AGENTS"
echo "Mean Safety Ratio (Learning): $MAICBF_MEAN_SAFETY_RATIO"
echo "Scalability (Time): $MAICBF_TIME"
echo "Number of agents outside input constraint: $NUM_OUTSIDE_CONSTRAINTS"

echo "MACBF Results:"
echo "Deadlocked Agents: $MACBF_DEADLOCKED_AGENTS"
echo "Mean Safety Ratio (Learning): $MACBF_MEAN_SAFETY_RATIO"
echo "Scalability (Time): $MACBF_TIME"
echo "Number of agents outside input constraint: $NUM_OUTSIDE_CONSTRAINTS_MACBF"

echo "GCBF Results:"
echo "Deadlock Avoidance (Reach Rate): $GCBF_DEADLOCK_AVOIDANCE"
echo "Mean Safety Ratio (Safe Rate): $GCBF_MEAN_SAFETY_RATIO"
echo "Scalability (Time): $GCBF_TIME"
