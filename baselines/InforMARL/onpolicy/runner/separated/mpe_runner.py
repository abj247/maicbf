import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio
from pyvirtualdisplay import Display


def _t2n(x):
    return x.detach().cpu().numpy()


class MPERunner(Runner):
    dt = 0.1

    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        #self.eval_envs = None  # Initialize eval_envs to None

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            action_norms = []
            deadlock_counts = []
            input_constraints_violations = []

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)
                
                #print(values)

                #print("actions_env:", actions_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                #print(infos)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                #print("infos:", infos)
                # deadlock_agents = set()
                # if step >= 5 and step <= 31:
                    
                #     for agent_id in range(self.num_agents):
                #         deadlock = False
                #         for t in range(step - 5, step):
                #             if np.all(infos[t][agent_id]["Agent_speed"] <= 0.01):
                #                 deadlock = True
                #                 break
                #         if deadlock:
                #             deadlock_agents.add(agent_id)
                #             deadlock_counts.append(len(deadlock_agents))
                    #print(deadlock_counts)

                # Calculate input constraints violations
                # input_constraints_violation = np.count_nonzero(
                #     [info[agent_id]["Agent_speed"] > 0.2 for info in infos for agent_id in range(self.num_agents)], axis=-1
                # )
                # input_constraints_violations.append(np.max(input_constraints_violation))
                # print(input_constraints_violation)

                # insert data into buffer
                self.insert(data)

            # Check for deadlock in sliding window of size 6
            

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(f"Time taken: {end - start:.3f} seconds")
                # print(f"\n Scenario: {self.all_args.scenario_name} "
                #     f"Algo: {self.algorithm_name} Exp: {self.experiment_name} "
                #     f"Updates: {episode}/{episodes} episodes, total num timesteps: "
                #     f"{total_num_steps}/{self.num_env_steps}, "
                #     f"FPS: {int(total_num_steps / (end - start))}.\n")

                env_infos = {}
                if self.env_name == "MPE":
                    avg_ep_rews = []
                    total_agents_in_collision = set()
                    violating_agents = 0
                    deadlock_agents = set()
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        dist_goals, time_to_goals, min_times_to_goal = [], [], []
                        idv_collisions, obst_collisions, wall_collisions = [], [], []
                        agent_speeds = []
                        # iterate through rollouts
                        for info in infos:
                            if "individual_reward" in info[agent_id].keys():
                                idv_rews.append(info[agent_id]["individual_reward"])
                            if "Dist_to_goal" in info[agent_id].keys():
                                dist_goals.append(info[agent_id]["Dist_to_goal"])
                            if "Time_req_to_goal" in info[agent_id].keys():
                                times = info[agent_id]["Time_req_to_goal"]
                                if times == -1:
                                    times = (
                                        self.all_args.episode_length * self.dt
                                    )  # NOTE: Hardcoding `dt`
                                time_to_goals.append(times)
                            if "Num_agent_collisions" in info[agent_id].keys():
                                idv_collisions.append(
                                    info[agent_id]["Num_agent_collisions"]
                                )
                                if info[agent_id]["Num_agent_collisions"] > 0:
                                    total_agents_in_collision.add(agent_id)
                            if "Num_obst_collisions" in info[agent_id].keys():
                                obst_collisions.append(
                                    info[agent_id]["Num_obst_collisions"]
                                )
                                if info[agent_id]["Num_obst_collisions"] > 0:
                                    total_agents_in_collision.add(agent_id)
                            if "Num_wall_collisions" in info[agent_id].keys():
                                wall_collisions.append(
                                    info[agent_id]["Num_wall_collisions"]
                                )
                                if info[agent_id]["Num_wall_collisions"] > 0:
                                    total_agents_in_collision.add(agent_id)
                            if "Min_time_to_goal" in info[agent_id].keys():
                                min_times_to_goal.append(
                                    info[agent_id]["Min_time_to_goal"]
                                )
                            if "Agent_speed" in info[agent_id].keys():
                                agent_speeds.append(
                                    info[agent_id]["Agent_speed"]
                                )
                            
                        mean_agent_speed = np.mean(agent_speeds)
                        if mean_agent_speed > 0.2:
                            violating_agents += 1

                        # Check for deadlock in sliding window of size 6
                        for agent_id in range(self.num_agents):
                            for window_start in range(len(agent_speeds) - 5):
                                if all(speed <= 0.01 for speed in agent_speeds[window_start:window_start + 6]):
                                    deadlock_agents.add(agent_id)
                                else:
                                    deadlock_agents.discard(agent_id)

                        agent_rew = f"agent{agent_id}/individual_rewards"
                        times = f"agent{agent_id}/time_to_goal"
                        dists = f"agent{agent_id}/dist_to_goal"
                        agent_col = f"agent{agent_id}/num_agent_collisions"
                        obst_col = f"agent{agent_id}/num_obstacle_collisions"
                        wall_col = f"agent{agent_id}/num_wall_collisions"
                        min_times = f"agent{agent_id}/min_time_to_goal"
                        agent_speed = f"agent{agent_id}/agent_speed"

                        env_infos[agent_rew] = idv_rews
                        env_infos[times] = time_to_goals
                        env_infos[min_times] = min_times_to_goal
                        env_infos[dists] = dist_goals
                        env_infos[agent_col] = idv_collisions
                        env_infos[obst_col] = obst_collisions
                        env_infos[wall_col] = wall_collisions
                        env_infos[agent_speed] = agent_speeds

                        avg_agent_ep_rew = (
                            np.mean(self.buffer[agent_id].rewards) * self.episode_length
                        )
                        train_infos[agent_id].update(
                            {"individual_rewards": np.mean(idv_rews)}
                        )
                        train_infos[agent_id].update(
                            {"average_episode_rewards": avg_agent_ep_rew}
                        )
                        avg_ep_rews.append(avg_agent_ep_rew)
                #avg_deadlocks = np.mean(deadlock_counts) if deadlock_counts else 0
                #print(deadlock_counts.shape)
                total_collisions = np.sum(idv_collisions) + np.sum(obst_collisions) + np.sum(wall_collisions)
                print(
                    f"Average episode rewards is {np.sum(avg_ep_rews):.3f} \t"
                    f"Total timesteps: {total_num_steps} \t "
                    f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f} \t"
                    f"Average time to goal: {np.mean(time_to_goals):.3f} \t"
                    f"Average agent collisions: {np.sum(idv_collisions):.3f} \t"
                    f"Average obstacle collisions: {np.sum(obst_collisions):.3f} \t"
                    f"Average wall collisions: {np.sum(wall_collisions):.3f} \t"
                    f"Total collisions: {total_collisions:.3f} \t"
                    f"Number of agents in deadlocks: {len(deadlock_agents)} \t"
                    f"Input constraints violations: {violating_agents} \t"
                    f"Average agent speed: {np.mean(agent_speeds)} \t"
                    f"Total number of agents in collision: {len(total_agents_in_collision)}"
                )
                # env_infos.update({'average_episode_rewards': np.sum(avg_ep_rews)})
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            #if episode % self.eval_interval == 0 and self.use_eval:
            #if self.eval_envs is None:
             # Initialize eval_envs if not already initialized
            #self.eval(total_num_steps)

            # render and save visualization
            #self.render(episode)   

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(
                        self.envs.action_space[agent_id].high[i] + 1
                    )[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(
                    np.eye(self.envs.action_space[agent_id].n)[action], 1
                )
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        
        # Check if eval_envs is None
        if self.eval_envs is None:
            raise AttributeError("eval_envs is not initialized. Please initialize eval_envs before calling eval().")
        
        eval_obs = self.eval_envs.reset()
        #print(eval_obs)

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        #print(eval_rnn_states)
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        #print(eval_masks)      
        eval_action_norms = []
        eval_deadlock_counts = []
        eval_input_constraints_violations = []

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )

                eval_action = eval_action.detach().cpu().numpy()
                #print(eval_action)
                # rearrange action
                if (
                    self.eval_envs.action_space[agent_id].__class__.__name__
                    == "MultiDiscrete"
                ):
                    #print(eval_action)
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(
                            self.eval_envs.action_space[agent_id].high[i] + 1
                        )[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate(
                                (eval_action_env, eval_uc_action_env), axis=1
                            )
                    print(eval_action_env)
                elif (
                    self.eval_envs.action_space[agent_id].__class__.__name__
                    == "Discrete"
                ):
                    eval_action_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1
                    )
                else:
                    raise NotImplementedError
                #print(eval_action_env)

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                #print(eval_rnn_states)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)
            #print(eval_actions_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(
                eval_actions_env
            )
            print(eval_infos)
            eval_episode_rewards.append(eval_rewards)

            # Calculate the norm of actions
            eval_action_norm = np.linalg.norm(eval_temp_actions_env, axis=-1)
            eval_action_norms.append(eval_action_norm)

            # Calculate input constraints violations
            eval_input_constraints_violation = np.count_nonzero(eval_action_norm > 0.2, axis=-1)
            eval_input_constraints_violations.append(np.max(eval_input_constraints_violation))
            print(eval_input_constraints_violations)

            # Check for deadlock in sliding window of size 6
            eval_deadlock_agents = set()
            if eval_step >= 5 and eval_step <= 31:
                for agent_id in range(self.num_agents):
                    eval_deadlock = False
                    for t in range(eval_step - 5, eval_step):
                        if np.all(eval_temp_actions_env[t][agent_id, :] <= 0.01):
                            eval_deadlock = True
                            break
                    if eval_deadlock:
                        eval_deadlock_agents.add(agent_id)
                        eval_deadlock_counts.append(len(eval_deadlock_agents))

                    print(eval_deadlock_counts)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(
                np.sum(eval_episode_rewards[:, :, agent_id], axis=0)
            )
            eval_train_infos.append(
                {"eval_average_episode_rewards": eval_average_episode_rewards}
            )
            print(
                f"Eval average episode rewards of agent_{agent_id}: "
                + str(eval_average_episode_rewards)
            )

        print(
            f"Eval average episode rewards is {np.sum(eval_average_episode_rewards):.3f} \t"
            f"Total timesteps: {total_num_steps} \t "
            f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f} \t"
            f"Eval average time to goal: {np.mean(eval_time_to_goals):.3f} \t"
            f"Eval average agent collisions: {np.sum(eval_idv_collisions):.3f} \t"
            f"Eval average obstacle collisions: {np.sum(eval_obst_collisions):.3f} \t"
            f"Eval number of agents in deadlocks: {np.max(eval_deadlock_counts)} \t"
            f"Eval number of deadlocks: {np.sum(eval_deadlock_counts)} \t"
            f"Eval input constraints violations: {np.max(eval_input_constraints_violations)}"
        )

        self.log_train(eval_train_infos, total_num_steps)

    # @torch.no_grad()
    # def render(self, episode):
    #     all_frames = []
    #     for ep in range(episode + 1):
    #         episode_rewards = []
    #         obs = self.envs.reset()
    #         if self.all_args.save_gifs:
    #             image = self.envs.render("rgb_array")[0][0]
    #             all_frames.append(image)

    #         rnn_states = np.zeros(
    #             (
    #                 self.n_rollout_threads,
    #                 self.num_agents,
    #                 self.recurrent_N,
    #                 self.hidden_size,
    #             ),
    #             dtype=np.float32,
    #         )
    #         masks = np.ones(
    #             (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
    #         )

    #         for step in range(self.episode_length):
    #             calc_start = time.time()

    #             temp_actions_env = []
    #             for agent_id in range(self.num_agents):
    #                 if not self.use_centralized_V:
    #                     share_obs = np.array(list(obs[:, agent_id]))
    #                 self.trainer[agent_id].prep_rollout()
    #                 action, rnn_state = self.trainer[agent_id].policy.act(
    #                     np.array(list(obs[:, agent_id])),
    #                     rnn_states[:, agent_id],
    #                     masks[:, agent_id],
    #                     deterministic=True,
    #                 )

    #                 action = action.detach().cpu().numpy()
    #                 # rearrange action
    #                 if (
    #                     self.envs.action_space[agent_id].__class__.__name__
    #                     == "MultiDiscrete"
    #                 ):
    #                     for i in range(self.envs.action_space[agent_id].shape):
    #                         uc_action_env = np.eye(
    #                             self.envs.action_space[agent_id].high[i] + 1
    #                         )[action[:, i]]
    #                         if i == 0:
    #                             action_env = uc_action_env
    #                         else:
    #                             action_env = np.concatenate(
    #                                 (action_env, uc_action_env), axis=1
    #                             )
    #                 elif (
    #                     self.envs.action_space[agent_id].__class__.__name__
    #                     == "Discrete"
    #                 ):
    #                     action_env = np.squeeze(
    #                         np.eye(self.envs.action_space[agent_id].n)[action], 1
    #                     )
    #                 else:
    #                     raise NotImplementedError

    #                 temp_actions_env.append(action_env)
    #                 rnn_states[:, agent_id] = _t2n(rnn_state)

    #             # [envs, agents, dim]
    #             actions_env = []
    #             for i in range(self.n_rollout_threads):
    #                 one_hot_action_env = []
    #                 for temp_action_env in temp_actions_env:
    #                     one_hot_action_env.append(temp_action_env[i])
    #                 actions_env.append(one_hot_action_env)

    #             # Obser reward and next obs
    #             obs, rewards, dones, infos = self.envs.step(actions_env)
    #             episode_rewards.append(rewards)

    #             rnn_states[dones == True] = np.zeros(
    #                 ((dones == True).sum(), self.recurrent_N, self.hidden_size),
    #                 dtype=np.float32,
    #             )
    #             masks = np.ones(
    #                 (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
    #             )
    #             masks[dones == True] = np.zeros(
    #                 ((dones == True).sum(), 1), dtype=np.float32
    #             )

    #             if self.all_args.save_gifs:
    #                 image = self.envs.render("rgb_array")[0][0]
    #                 all_frames.append(image)
    #                 calc_end = time.time()
    #                 elapsed = calc_end - calc_start
    #                 if elapsed < self.all_args.ifi:
    #                     time.sleep(self.all_args.ifi - elapsed)

    #         episode_rewards = np.array(episode_rewards)
    #         for agent_id in range(self.num_agents):
    #             average_episode_rewards = np.mean(
    #                 np.sum(episode_rewards[:, :, agent_id], axis=0)
    #             )
    #             print(
    #                 "eval average episode rewards of agent%i: " % agent_id
    #                 + str(average_episode_rewards)
    #             )

    #     if self.all_args.save_gifs:
    #         imageio.mimsave(
    #             str(self.gif_dir) + f"/render_episode_{episode}.gif",
    #             all_frames,
    #             duration=self.all_args.ifi,
    #         )

    @torch.no_grad()
    def render(self, episode):
        # Start a virtual display
        display = Display(visible=0, size=(1400, 900), backend="xvfbwrapper")
        display.start()

        all_frames = []
        for ep in range(episode + 1):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render("rgb_array")[0][0]
                all_frames.append(image)

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(obs[:, agent_id])),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,
                    )

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if (
                        self.envs.action_space[agent_id].__class__.__name__
                        == "MultiDiscrete"
                    ):
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(
                                self.envs.action_space[agent_id].high[i] + 1
                            )[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate(
                                    (action_env, uc_action_env), axis=1
                                )
                    elif (
                        self.envs.action_space[agent_id].__class__.__name__
                        == "Discrete"
                    ):
                        action_env = np.squeeze(
                            np.eye(self.envs.action_space[agent_id].n)[action], 1
                        )
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if self.all_args.save_gifs:
                    image = self.envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(
                    np.sum(episode_rewards[:, :, agent_id], axis=0)
                )
                print(
                    "eval average episode rewards of agent%i: " % agent_id
                    + str(average_episode_rewards)
                )

        if self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + f"/render_episode_{episode}.gif",
                all_frames,
                duration=self.all_args.ifi,
            )

        # Stop the virtual display
        display.stop()
