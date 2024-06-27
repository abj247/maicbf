#!/bin/bash

# Define variables
NUM_AGENTS=<num_agents>
MODEL_PATH=<model_path>

# Run the first script (maicbf)
echo "Running MAICBF evaluation..."
MAICBF_OUTPUT=$(python eval.py --num agents $NUM_AGENTS --model_path $MODEL_PATH)

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
