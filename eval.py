import sys
sys.dont_write_bytecode = True

import os
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import pandas as pd
import core
import config
from plot import PlotHelper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ref', type=str, default=None)
    args = parser.parse_args()
    return args


def build_evaluation_graph(num_agents):
    # s is the state vectors of the agents
    s = tf.placeholder(tf.float32, [num_agents, 8])
    # s_ref is the goal states
    s_ref = tf.placeholder(tf.float32, [num_agents, 8])
    # x is difference between the state of each agent and other agents
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)
    # h is the CBF value of shape [num_agents, TOP_K, 1], where TOP_K represents
    # the K nearest agents
    h, mask, indices = core.network_cbf(
        x=x, r=config.DIST_MIN_THRES, indices=None)
    # u is the control action of each agent, with shape [num_agents, 3]
    u = core.network_action(
        s=s, s_ref=s_ref, obs_radius=config.OBS_RADIUS, indices=indices)
    safe_mask = core.compute_safe_mask(s, r=config.DIST_SAFE, indices=indices)
    # check if each agent is safe
    is_safe = tf.equal(tf.reduce_mean(tf.cast(safe_mask, tf.float32)), 1)

    # u_res is delta u. when u does not satisfy the CBF conditions, we want to compute
    # a u_res such that u + u_res satisfies the CBF conditions
    u_res = tf.Variable(tf.zeros_like(u), name='u_res')
    loop_count = tf.Variable(0, name='loop_count')
   
    def opt_body(u_res, loop_count, is_safe):
        # a loop of updating u_res
        # compute s_next under u + u_res
        dsdt = core.quadrotor_dynamics_tf(s, u + u_res)
        s_next = s + dsdt * config.TIME_STEP_EVAL
        x_next = tf.expand_dims(s_next, 1) - tf.expand_dims(s_next, 0)
        h_next, mask_next, _ = core.network_cbf(
            x=x_next, r=config.DIST_MIN_THRES, indices=indices)
        # deriv should be >= 0. if not, we update u_res by gradient descent
        deriv = h_next - h + config.TIME_STEP_EVAL * config.ALPHA_CBF * h
        deriv = deriv * mask * mask_next
        error = tf.reduce_sum(tf.math.maximum(-deriv, 0), axis=1)\
        # compute the gradient to update u_res
        error_gradient = tf.gradients(error, u_res)[0]
        u_res = u_res - config.REFINE_LEARNING_RATE * error_gradient
        loop_count = loop_count + 1
        return u_res, loop_count, is_safe

    def opt_cond(u_res, loop_count, is_safe):
        # update u_res for REFINE_LOOPS
        cond = tf.logical_and(
            tf.less(loop_count, config.REFINE_LOOPS), 
            tf.logical_not(is_safe))
        return cond
    
    with tf.control_dependencies([
        u_res.assign(tf.zeros_like(u)), loop_count.assign(0)]):
        u_res, _, _ = tf.while_loop(opt_cond, opt_body, [u_res, loop_count, is_safe])
        u_opt = u + u_res

    # compute the value of loss functions and the accuracies
    # loss_dang is for h(s) < 0, s in dangerous set
    # loss safe is for h(s) >=0, s in safe set
    # acc_dang is the accuracy that h(s) < 0, s in dangerous set is satisfied
    # acc_safe is the accuracy that h(s) >=0, s in safe set is satisfied
    loss_dang, loss_safe, acc_dang, acc_safe = core.loss_barrier(
        h=h, s=s, indices=indices)
    # loss_dang_deriv is for doth(s) + alpha h(s) >=0 for s in dangerous set
    # loss_safe_deriv is for doth(s) + alpha h(s) >=0 for s in safe set
    # loss_medium_deriv is for doth(s) + alpha h(s) >=0 for s not in the dangerous
    # or the safe set
    (loss_dang_deriv, loss_safe_deriv, loss_medium_deriv, acc_dang_deriv, 
    acc_safe_deriv, acc_medium_deriv) = core.loss_derivatives(
        s=s, u=u_opt, h=h, x=x, indices=indices)
    # the distance between the u_opt and the nominal u
    loss_action = core.loss_actions(s=s, u=u_opt, s_ref=s_ref, indices=indices)
    loss_dang_ic, loss_safe_ic, acc_dang_ic, acc_safe_ic = core.loss_barrier_ic(
        u=u, indices=indices)
    loss_agile = core.loss_agile(s=s, s_ref=s_ref, u=u, v_max=0.2, sigma_tight=0.05)

    loss_list = [loss_dang, loss_safe, loss_dang_deriv, 
                 loss_safe_deriv, loss_medium_deriv, loss_action, loss_dang_ic,loss_safe_ic,loss_agile]
    acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv, acc_medium_deriv, acc_dang_ic,acc_safe_ic]

    return s, s_ref, u_opt, loss_list, acc_list

    
def print_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_i = acc[:, i]
        acc_list.append(np.mean(acc_i[acc_i > 0]))
    print('Accuracy: {}'.format(acc_list))


def render_init(num_agents):
    fig = plt.figure(figsize=(10, 7))
    return fig


def show_obstacles(obs, ax, z=[0, 6], alpha=0.6, color='deepskyblue'):
    for x1, y1, x2, y2 in obs:
        xs, ys = np.meshgrid([x1, x2], [y1, y2])
        zs = np.ones_like(xs)
        ax.plot_surface(xs, ys, zs * z[0], alpha=alpha, color=color)
        ax.plot_surface(xs, ys, zs * z[1], alpha=alpha, color=color)

        xs, zs = np.meshgrid([x1, x2], z)
        ys = np.ones_like(xs)
        ax.plot_surface(xs, ys * y1, zs, alpha=alpha, color=color)
        ax.plot_surface(xs, ys * y2, zs, alpha=alpha, color=color)

        ys, zs = np.meshgrid([y1, y2], z)
        xs = np.ones_like(ys)
        ax.plot_surface(xs * x1, ys, zs, alpha=alpha, color=color)
        ax.plot_surface(xs * x2, ys, zs, alpha=alpha, color=color)


def clip_norm(x, thres):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    mask = (norm > thres).astype(np.float32)
    x = x * (1 - mask) + x * mask / (1e-6 + norm)
    return x


def clip_state(s, x_thres, v_thres=0.1, h_thres=6):
    x, v, r = s[:, :3], s[:, 3:6], s[:, 6:]
    x = np.concatenate([np.clip(x[:, :2], 0, x_thres),
                        np.clip(x[:, 2:], 0, h_thres)], axis=1)
    v = clip_norm(v, v_thres)
    s = np.concatenate([x, v, r], axis=1)
    return s


def main():
  
    args = parse_args()
    s, s_ref, u, loss_list, acc_list = build_evaluation_graph(args.num_agents)
    # loads the pretrained weights
    vars = tf.trainable_variables()
    vars_restore = []
    for v in vars:
        if 'action' in v.name or 'cbf' in v.name:
            vars_restore.append(v)
    # initialize the tensorflow Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=vars_restore)
    saver.restore(sess, args.model_path)

    safety_ratios_epoch = []
    safety_ratios_epoch_baseline = []

    dist_errors = []
    dist_errors_baseline = []
    accuracy_lists = []

    if args.vis > 0:
        plt.ion()
        plt.close()
        fig = render_init(args.num_agents)
    # initialize the environment
    scene = core.Maze(args.num_agents, max_steps=args.max_steps)
    if args.ref is not None:
        scene.read(args.ref)

    if not os.path.exists('trajectory'):
        os.mkdir('trajectory')
    traj_dict = {'ours': [], 'baseline': [], 'obstacles': [np.array(scene.OBSTACLES)]}
    total_steps = config.EVALUATE_STEPS * args.max_steps  # This should correctly reflect the product
    collision_tracking = np.zeros((total_steps, args.num_agents), dtype=int)

    safety_reward = []
    dist_reward = []
    u_values= []
    current_step_index = 0
    for istep in range(config.EVALUATE_STEPS):
        if args.vis > 0:
            plt.clf()
            ax_1 = fig.add_subplot(121, projection='3d')
            ax_2 = fig.add_subplot(122, projection='3d')
        safety_ours = []
        safety_baseline = []

        scene.reset()
        start_time = time.time()
        s_np = np.concatenate(
            [scene.start_points, np.zeros((args.num_agents, 5))], axis=1)
        s_traj = []
        safety_info = np.zeros(args.num_agents, dtype=np.float32)
        # a scene has a sequence of goal states for each agent. in each scene.step,
        # we move to a new goal state
        print(scene.steps)
        for t in range(scene.steps):
            # the goal states
            s_ref_np = np.concatenate(
                [scene.waypoints[t], np.zeros((args.num_agents, 5))], axis=1)
            # run INNER_LOOPS_EVAL steps to reach the goal state
            
            for i in range(config.INNER_LOOPS_EVAL):
                u_np, acc_list_np = sess.run(
                    [u, acc_list], feed_dict={s:s_np, s_ref: s_ref_np})
                
                if args.vis == 1:
                    u_ref_np = core.quadrotor_controller_np(s_np, s_ref_np)
                    u_np = clip_norm(u_np - u_ref_np, 100.0) + u_ref_np
                dsdt = core.quadrotor_dynamics_np(s_np, u_np)
                s_np = s_np + dsdt * config.TIME_STEP_EVAL
                safety_ratio = 1 - np.mean(
                    core.dangerous_mask_np(s_np, config.DIST_MIN_CHECK), axis=1)
                individual_safety = safety_ratio == 1
                safety_ours.append(individual_safety)
                safety_info = safety_info + individual_safety - 1
                safety_ratio = np.mean(individual_safety)
                safety_ratios_epoch.append(safety_ratio)
                accuracy_lists.append(acc_list_np)
                if np.mean(
                    np.linalg.norm(s_np[:, :3] - s_ref_np[:, :3], axis=1)
                    ) < config.DIST_TOLERATE:
                    break
            
                #print(u_values)
                s_traj.append(np.expand_dims(s_np[:, [0, 1, 2, 6, 7]], axis=0))
            collisions = core.dangerous_mask_np(s_np, config.DIST_MIN_CHECK)
            collision_tracking[current_step_index] = np.any(collisions, axis=1).astype(int)

            current_step_index += 1  # Move to the next global step
            u_values.append(u_np.copy())
        safety_reward.append(np.mean(safety_info))
        dist_reward.append(np.mean((np.linalg.norm(
            s_np[:, :3] - s_ref_np[:, :3], axis=1) < 1.5).astype(np.float32) * 10))
        dist_errors.append(
            np.mean(np.linalg.norm(s_np[:, :3] - s_ref_np[:, :3], axis=1)))
        

        traj_dict['ours'].append(np.concatenate(s_traj, axis=0))
        end_time = time.time()

        # reach the same goals using LQR controller without considering the collision
        s_np = np.concatenate(
            [scene.start_points, np.zeros((args.num_agents, 5))], axis=1)
        s_traj = []
        for t in range(scene.steps):
            s_ref_np = np.concatenate(
                [scene.waypoints[t], np.zeros((args.num_agents, 5))], axis=1)
            for i in range(config.INNER_LOOPS_EVAL):
                u_np = core.quadrotor_controller_np(s_np, s_ref_np)
                dsdt = core.quadrotor_dynamics_np(s_np, u_np)
                s_np = s_np + dsdt * config.TIME_STEP_EVAL
                safety_ratio = 1 - np.mean(
                    core.dangerous_mask_np(s_np, config.DIST_MIN_CHECK), axis=1)
                individual_safety = safety_ratio == 1
                safety_baseline.append(individual_safety)
                safety_ratio = np.mean(individual_safety)
                safety_ratios_epoch_baseline.append(safety_ratio)
                s_traj.append(np.expand_dims(s_np[:, [0, 1, 2, 6, 7]], axis=0))
        dist_errors_baseline.append(np.mean(np.linalg.norm(s_np[:, :3] - s_ref_np[:, :3], axis=1)))
        traj_dict['baseline'].append(np.concatenate(s_traj, axis=0))
        
        if args.vis > 0:
            # visualize the trajectories
            s_traj_ours = traj_dict['ours'][-1]
            s_traj_baseline = traj_dict['baseline'][-1]
    
            for j in range(0, max(s_traj_ours.shape[0], s_traj_baseline.shape[0]), 10):
                ax_1.clear()
                ax_1.view_init(elev=80, azim=-45)
                ax_1.axis('off')
                show_obstacles(scene.OBSTACLES, ax_1)
                j_ours = min(j, s_traj_ours.shape[0]-1)
                s_np = s_traj_ours[j_ours]
                safety = safety_ours[j_ours]

                ax_1.set_xlim(0, 20)
                ax_1.set_ylim(0, 20)
                ax_1.set_zlim(0, 10)
                ax_1.scatter(s_np[:, 0], s_np[:, 1], s_np[0, 2], 
                             color='darkorange', label='Agent')
                ax_1.scatter(s_np[safety<1, 0], s_np[safety<1, 1], s_np[safety<1, 2], 
                             color='red', label='Collision')
                ax_1.set_title('Ours: Safety Rate = {:.4f}'.format(
                    np.mean(safety_ratios_epoch)), fontsize=16)

                ax_2.clear()
                ax_2.view_init(elev=80, azim=-45)
                ax_2.axis('off')
                show_obstacles(scene.OBSTACLES, ax_2)
                j_baseline = min(j, s_traj_baseline.shape[0]-1)
                s_np = s_traj_baseline[j_baseline]
                safety = safety_baseline[j_baseline]

                ax_2.set_xlim(0, 20)
                ax_2.set_ylim(0, 20)
                ax_2.set_zlim(0, 10)
                ax_2.scatter(s_np[:, 0], s_np[:, 1], s_np[1, 2], 
                             color='darkorange', label='Agent')
                ax_2.scatter(s_np[safety<1, 0], s_np[safety<1, 1], s_np[safety<1, 2], 
                             color='red', label='Collision')
                ax_2.set_title('LQR: Safety Rate = {:.4f}'.format(
                    np.mean(safety_ratios_epoch_baseline)), fontsize=16)
                plt.legend(loc='lower right')

                fig.canvas.draw()
                plt.pause(0.001)
                
       
        
        print('Evaluation Step: {} | {}, Time: {:.4f}'.format(
            istep + 1, config.EVALUATE_STEPS, end_time - start_time))
    

    collision_tracking = np.clip(collision_tracking, 0, 1)
    base_directory = 'csv_data'
    sub_directory = 'collision_tracking'
    directory_path = os.path.join(base_directory, sub_directory)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Define the full path to the CSV file within the subdirectory
    #csv_file_path = os.path.join(directory_path, 'hu_time_umax_0.2_agile_weight_0.5_64_agents.csv')
    csv_file_path = os.path.join(directory_path, 'hu_time_baseline_4_agents_itr_03.csv')

    ## Export to CSV, taking into account the new shape of collision_tracking
    column_names = ['Agent {}'.format(i+1) for i in range(args.num_agents)]
    df_collision_tracking = pd.DataFrame(collision_tracking, columns=column_names)
    df_collision_tracking.to_csv(csv_file_path, index_label="Step")
    print("collision tracking data saved!!!")

    print_accuracy(accuracy_lists)
    print('Distance Error (Learning | Baseline): {:.4f} | {:.4f}'.format(
          np.mean(dist_errors), np.mean(dist_errors_baseline)))
    print('Mean Safety Ratio (Learning | Baseline): {:.4f} | {:.4f}'.format(
          np.mean(safety_ratios_epoch), np.mean(safety_ratios_epoch_baseline)))

    safety_reward = np.mean(safety_reward)
    dist_reward = np.mean(dist_reward)
    print('Safety Reward: {:.4f}, Dist Reward: {:.4f}, Reward: {:.4f}'.format(
        safety_reward, dist_reward, 9 + 0.1 * (safety_reward + dist_reward)))

    pickle.dump(traj_dict, open('trajectory/traj_eval.pkl', 'wb'))
    scene.write_trajectory('trajectory/env_traj_eval.pkl', traj_dict['ours'])

    # control_input_1 = [u_step[0, 0] for u_step in u_values]  
    # control_input_2 = [u_step[0, 1] for u_step in u_values]  
    # control_input_3 = [u_step[0, 2] for u_step in u_values]  
   
    
    
    # time_steps = list(range(len(u_values)))

    u_max = 0.6
    u_max_squared = u_max**2

    # Assuming u_values is structured with each element as [angular_velocity_x, angular_velocity_y, linear_acceleration]
    # Calculate squared values for each control input
    control_input_1_squared = [u_max_squared- (u_step[0, 0]**2) for u_step in u_values]  
    control_input_2_squared = [u_max_squared-(u_step[0, 1]**2) for u_step in u_values]  
    control_input_3_squared = [u_max_squared-(u_step[0, 2]**2) for u_step in u_values] 
    
    #modified_control_inputs = [u_max_squared - (u_step[0, 0]**2 + u_step[0, 1]**2 + u_step[0, 2]**2) for u_step in u_values]
    modified_control_inputs = [u_max_squared - (u_step[:, 0]**2 + u_step[:, 1]**2 + u_step[:, 2]**2) for u_step in u_values]
    exceed_threshold_count_v = 0
    exceed_threshold_count_a = 0

    a_values_np = []
    v_values_np = []

    for u_step in u_values:
        a_value = u_step[:, 2]
        v_value = np.sqrt(u_step[:, 0]**2 + u_step[:, 1]**2)
        if np.any(np.abs(a_value) > 2):
            exceed_threshold_count_a += 1
            a_value[np.abs(a_value) > 2] = 0
        if np.any(v_value > 2):
            exceed_threshold_count_v += 1
            v_value[v_value > 2] = 0

        a_values_np.append(a_value)
        v_values_np.append(v_value)

    v_values = np.array(v_values_np)
    max_v_values_all_agents = np.max(v_values, axis=0)
    a_values = np.array(a_values_np)
    #modified_control_inputs = [u_max_squared - (u_step[0]**2) for u_step in u_values]
    modified_control_inputs_array = np.array(modified_control_inputs)
    log_modified_control_inputs = np.log(1 + modified_control_inputs_array )

    # To avoid invalid values for log, ensure all values are positive
    #safe_modified_control_inputs_array = np.maximum(0, modified_control_inputs_array) + 1e-6
    #log_modified_control_inputs = np.log(1 + safe_modified_control_inputs_array)


    time_steps = list(range(len(u_values)))
    u_values_array = np.array(u_values)
    #print(u_values.shape)

    # Creating a DataFrame
    df = pd.DataFrame({
        'Time Steps': time_steps,
        'h': modified_control_inputs
    })

    #csv logging

    

    # Specify your desired path to save the CSV file
    #csv_file_path = 'csv_data/hu_data/hu_time_umax_0.2_agile_weight_0.5_64_agents.csv'
    csv_file_path = 'csv_data/hu_time_baseline_itr_03_4_agents.csv'

    # Save the DataFrame to a CSV file
    #PlotHelper.save_to_csv(time_steps, modified_control_inputs, csv_file_path)
    df.to_csv(csv_file_path, index=False)

    print(f"CSV file has been saved to {csv_file_path}")

    # Plotting

    #PlotHelper.plot_data(time_steps, modified_control_inputs_array, 'Time Steps', 'h(u)', 'h(u) for all agents (baseline)_4 agents)', 'agents', 'h(u)_baseline_all_agents_4.png')


    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, modified_control_inputs_array, label='agents')
    # plt.plot(time_steps, u_values_array[:,0,1], label='h_u for agent 0')
    # plt.plot(time_steps, u_values_array[:,0,2], label='h_u for agent 0')
    plt.xlabel('Time Steps')
    plt.ylabel('h(u)')
    plt.title('h(u) for all agents (baseline)_4 agents)')
    plt.legend()
    plt.savefig('h(u)_baseline_all_agents_4_itr_03.png', dpi=300)
    plt.show()


   #Plot Acceleration
    #PlotHelper.plot_data(time_steps, a_values, 'Time Steps', 'acceleration', 'acceleration for all agents (baseline)_4 agents)', 'acceleration', 'acc_baseline_all_agents_4.png')

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, a_values, label='acceleration')
    # plt.plot(time_steps, u_values_array[:,0,1], label='h_u for agent 0')
    # plt.plot(time_steps, u_values_array[:,0,2], label='h_u for agent 0')
    plt.xlabel('Time Steps')
    plt.ylabel('acceleration')
    plt.title('acceleration for all agents (baseline)_4 agents)')
    plt.legend()
    plt.savefig('acc_baseline_all_agents_4_itr_03.png', dpi=300)
    plt.show()


    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, v_values, label='velocity')
    plt.xlabel('Time Steps')
    plt.ylabel('velocity')
    plt.title('velocity for all agents_baseline')
    plt.legend()
    plt.savefig('velocity_baseline_all_agents_itr_03.png', dpi=300)
    plt.show()

    print(max_v_values_all_agents)
    print(exceed_threshold_count_a)
    print( exceed_threshold_count_v)

    # using plot.py

    #time_steps = list(range(len(u_values)))

    # # Plot for Squared Control Input 1
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_steps, control_input_1_squared, label='h_u (w_x) for agent 0')
    # plt.axhline(y=u_max_squared, color='r', linestyle='-', label='$u_{max}^2$')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Squared Control Input Value')
    # plt.title('Squared Control Input 1 Over Time for Agent 0')
    # plt.legend()
    # plt.savefig('squared_control_input_1_agent_0_old.png', dpi=300)
    # plt.show()

    # # Plot for Squared Control Input 2
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_steps, control_input_2_squared, label='h_u (w_y) for agent 0')
    # plt.axhline(y=u_max_squared, color='r', linestyle='-', label='$u_{max}^2$')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Squared Control Input Value')
    # plt.title('Squared Control Input 2 Over Time for Agent 0 ')
    # plt.legend()
    # plt.savefig('squared_control_input_2_agent_0_old.png', dpi=300)
    # plt.show()

    # # Plot for Squared Control Input 3
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_steps, control_input_3_squared, label='h_u (a) for Agent 0')
    # plt.axhline(y=u_max_squared, color='r', linestyle='-', label='$u_{max}^2$')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Squared Control Input Value')
    # plt.title('Squared Control Input 3 Over Time for Agent 0')
    # plt.legend()
    # plt.savefig('squared_control_input_3_agent_0_old.png', dpi=300)
    # plt.show()

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_steps, control_input_1, label='Control Input 1 for Agent 0')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Control Input Value')
    # plt.title('Control Input 1 Over Time for Agent 0')
    # plt.legend()
    # plt.savefig('control_input_1_agent_0.png', dpi=300)
    # plt.show()

    # # Plot for Control Input 2
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_steps, control_input_2, label='Control Input 2 for Agent 0')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Control Input Value')
    # plt.title('Control Input 2 Over Time for Agent 0')
    # plt.legend()
    # plt.savefig('control_input_2_agent_0.png', dpi=300)
    # plt.show()

    # # Plot for Control Input 3
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_steps, control_input_3, label='Control Input 3 for Agent 0')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Control Input Value')
    # plt.title('Control Input 3 Over Time for Agent 0')
    # plt.legend()
    # plt.savefig('control_input_3_agent_0.png', dpi=300)
    # plt.show()


# u_opt_history=[]
# u_opt_history = np.array(u_opt_history)
# print(u_opt_history.shape)
# # Plot each component of u_opt
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.plot(u_opt_history[:, 0])
# plt.title('Control Action 1')
# plt.xlabel('Time Step')
# plt.ylabel('Value')

# plt.subplot(1, 3, 2)
# plt.plot(u_opt_history[:, 1])
# plt.title('Control Action 2')
# plt.xlabel('Time Step')
# plt.ylabel('Value')

# plt.subplot(1, 3, 3)
# plt.plot(u_opt_history[:, 2])
# plt.title('Control Action 3')
# plt.xlabel('Time Step')
# plt.ylabel('Value')

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 8))

# for i in range(4):
#     plt.subplot(2, 2, i + 1)
#     plt.plot(u_opt_history[:, i])
#     plt.title(f'Control Action {i+1}')
#     plt.xlabel('Time Step')
#     plt.ylabel('Value')

# plt.tight_layout()
# plt.show()

if __name__ == '__main__':
    main()
