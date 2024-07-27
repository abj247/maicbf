import argparse
from utils import make_env, dict2csv
import numpy as np
import contextlib
import torch
from ckpt_plot.plot_curve import plot_result
import os
import time
from tqdm import tqdm  


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def eval_model_q(args):
    plot = {'good_rewards': [], 'adversary_rewards': [], 'rewards': [], 'steps': [], 'q_loss': [], 'gcn_q_loss': [],
            'p_loss': [], 'final': [], 'abs': []}
    best_eval_reward = -100000000

    print('=================== start eval ===================')
    eval_env = make_env(args.scenario, args)
    eval_env.seed(args.seed + 10)
    eval_rewards = []
    good_eval_rewards = []

    checkpoint = torch.load(args.checkpoint_path)
    agent = checkpoint['agents']
    info_n = []
    start_time = time.time()  # Start time for the evaluation process
    with temp_seed(args.seed):
        for n_eval in tqdm(range(args.num_eval_runs), desc="Evaluation Progress"):  # Add progress bar
            obs_n = eval_env.reset()
            episode_reward = 0
            episode_step = 0
            n_agents = eval_env.n
            agents_rew = [[] for _ in range(n_agents)]
            while True:
                action_n = agent.select_action(torch.Tensor(obs_n), action_noise=True,
                                                                                             param_noise=False).squeeze().cpu().numpy()
                next_obs_n, reward_n, done_n, infos = eval_env.step(action_n)
                info_n.append(infos)
                episode_step += 1
                terminal = (episode_step >= args.num_steps)
                episode_reward += np.sum(reward_n)
                for i, r in enumerate(reward_n):
                    agents_rew[i].append(r)
                obs_n = next_obs_n
                if done_n[0] or terminal:
                    eval_rewards.append(episode_reward)
                    agents_rew = [np.sum(rew) for rew in agents_rew]
                    good_reward = np.sum(agents_rew)
                    good_eval_rewards.append(good_reward)
                    if n_eval % 100 == 0:
                        print('test reward', episode_reward)
                    break

        info_n = np.array(info_n)
        print(info_n.shape)
        
        total_agent_collisions = []
        total_wall_collisions = []
        agents_in_collision = []
        violating_agents = []
        deadlock_agents = set()
        num_deadlocks = 0

        agent_speeds = [[] for _ in range(eval_env.n)]
        window_size = 6

        for step_info in info_n:
            step_collisions = 0
            step_agents_in_collision = set()
            step_violating_agents = set()
            step_wall_collisions = 0
            for agent_id, info in enumerate(step_info):
                if "Num_agent_collisions" in info.keys():
                    step_collisions += info["Num_agent_collisions"]
                    if info["Num_agent_collisions"] > 0:
                        step_agents_in_collision.add(agent_id)
                if "Num_wall_collisions" in info.keys():
                    step_wall_collisions += info["Num_wall_collisions"]
                if "Agent_speed" in info.keys():
                    agent_speeds[agent_id].append(info["Agent_speed"])
                    if info["Agent_speed"] > 0.2:
                        step_violating_agents.add(agent_id)
            total_agent_collisions.append(step_collisions)
            total_wall_collisions.append(step_wall_collisions)
            agents_in_collision.append(len(step_agents_in_collision))
            violating_agents.append(len(step_violating_agents))

            # Check for deadlock in sliding window of size 6
            if len(agent_speeds[0]) >= window_size:
                for agent_id in range(eval_env.n):
                    if all(speed <= 0.01 for speed in agent_speeds[agent_id][-window_size:]):
                        deadlock_agents.add(agent_id)
                    else:
                        deadlock_agents.discard(agent_id)

        num_deadlocks = len(deadlock_agents)

        avg_total_agent_collisions = np.mean(total_agent_collisions)
        avg_total_wall_collisions = np.mean(total_wall_collisions)
        avg_agents_in_collision = np.mean(agents_in_collision)
        avg_violating_agents = np.mean(violating_agents)
        avg_agent_speed = np.mean([np.mean(speeds) for speeds in agent_speeds if speeds])  # Calculate average speed
        print(f"Average total agent collisions: {int(avg_total_agent_collisions/2)}")
        print(f"Average total wall collisions: {int(avg_total_wall_collisions/2)}")
        print(f"Average number of agents in collision: {int(avg_agents_in_collision)}")
        print(f"Average number of violating agents: {int(avg_violating_agents)}")
        print(f"Number of agents in deadlock: {len(deadlock_agents)}")
        print(f"Number of deadlocks: {num_deadlocks}")
        print(f"Average agent speed: {avg_agent_speed:.2f}")  # Log average speed

        if np.mean(eval_rewards) > best_eval_reward:
            best_eval_reward = np.mean(eval_rewards)
            torch.save({'agents': agent}, os.path.join(args.save_dir, 'agents_best.ckpt'))

        plot['rewards'].append(np.mean(eval_rewards))
        plot['steps'].append(0)  # No training steps in evaluation
        plot['q_loss'].append(0)  # No q_loss in evaluation
        plot['p_loss'].append(0)  # No p_loss in evaluation
        print("========================================================")
        print(
            "Total eval runs: {}, total time: {} s".
                format(args.num_eval_runs, time.time() - args.start_time))
        print("GOOD reward: avg {} std {}, average reward: {}, best reward {}".format(np.mean(eval_rewards),
                                                                                      np.std(eval_rewards),
                                                                                      np.mean(plot['rewards'][-10:]),
                                                                                      best_eval_reward))
        plot['final'].append(np.mean(plot['rewards'][-10:]))
        plot['abs'].append(best_eval_reward)
        dict2csv(plot, os.path.join(args.save_dir, 'eval_curve.csv'))
        eval_env.close()
    total_time_elapsed = time.time() - start_time  # Calculate total time elapsed
    print(f"Total time elapsed for the whole evaluation process: {total_time_elapsed:.2f} seconds")  # Print total time elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Evaluation example')
    parser.add_argument('--scenario', required=True,
                        help='name of the environment to run')
    parser.add_argument('--seed', type=int, default=9, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--num_eval_runs', type=int, default=1, help='number of runs per evaluation (default: 5)')
    parser.add_argument('--num_steps', type=int, default=25, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument("--checkpoint_path", type=str, default="./ckpt_plot/coop_navigation_n6/agents_best.ckpt", required=True, help="path to the checkpoint file")
    parser.add_argument("--save_dir", type=str, default="./ckpt_plot",
                        help="directory in which evaluation results should be saved")
    args = parser.parse_args()
    args.start_time = time.time()

    eval_model_q(args)