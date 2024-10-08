import sys
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import os
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import core
import config

import cvxpy as cp
import do_mpc
import casadi as cs
import imageio




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, default=64)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--ref', type=str, default=None)
    parser.add_argument('--env', type=str, choices=['Empty', 'Maze'], default='Maze')
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

def show_obstacles(obs, ax, z=[0, 6], alpha=1.0, color= '#c69874'):  # Wooden brown color
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
    deadlock_ratios_epoch = []  # For tracking deadlock ratios
    deadlock_ratios_epoch_baseline = []

    dist_errors = []
    dist_errors_baseline = []
    accuracy_lists = []

    # Initialize a list to track steps taken by each agent to reach the goal for each evaluation step
    steps_taken_by_agents = np.zeros((args.num_agents, config.EVALUATE_STEPS))

    if args.vis > 0:
        plt.ion()
        plt.close()
        fig = render_init(args.num_agents)
    # initialize the environment
    scene = core.Empty(args.num_agents, max_steps=args.max_steps) if args.env == 'Empty' else core.Maze(args.num_agents, max_steps=args.max_steps)
    if args.ref is not None:
        scene.read(args.ref)

    if not os.path.exists('trajectory'):
        os.mkdir('trajectory')
    traj_dict = {'ours': [], 'baseline': []}
    total_steps = config.EVALUATE_STEPS * args.max_steps  
    collision_tracking = np.zeros((total_steps, args.num_agents), dtype=int)
    deadlock_tracking = np.zeros((total_steps, args.num_agents), dtype=int)  

    safety_reward = []
    dist_reward = []
    u_values= []
   
    for istep in range(config.EVALUATE_STEPS):
        safety_ours = []
        safety_baseline = []
        deadlock_ours = []  
        deadlock_baseline = [] 

        scene.reset()
        start_time = time.time()
        s_np = np.concatenate(
            [scene.start_points, np.zeros((args.num_agents, 5))], axis=1)
        s_traj = []
        safety_info = np.zeros(args.num_agents, dtype=np.float32)
        deadlock_info = np.zeros(args.num_agents, dtype=np.float32) 

        print(scene.steps)
        for t in range(scene.steps):
            s_ref_np = np.concatenate(
                [scene.waypoints[t], np.zeros((args.num_agents, 5))], axis=1)
            for i in range(config.INNER_LOOPS_EVAL):
                u_np, acc_list_np = sess.run(
                    [u, acc_list], feed_dict={s:s_np, s_ref: s_ref_np})
                
                if args.vis == 1:
                    u_ref_np = core.quadrotor_controller_np(s_np, s_ref_np)
                    u_np = clip_norm(u_np - u_ref_np, 100.0) + u_ref_np
                dsdt = core.quadrotor_dynamics_np(s_np, u_np)
                u_ref_np = core.quadrotor_controller_np(s_np, s_ref_np)

                s_np = s_np + dsdt * config.TIME_STEP_EVAL
                safety_ratio = 1 - np.mean(
                    core.dangerous_mask_np(s_np, config.DIST_MIN_CHECK), axis=1)
                individual_safety = safety_ratio == 1
                safety_ours.append(individual_safety)
                safety_info = safety_info + individual_safety - 1
                safety_ratio = np.mean(individual_safety)
                safety_ratios_epoch.append(safety_ratio)
                accuracy_lists.append(acc_list_np)
                
                # Deadlock prevention logic
                gamma = 0.99
                lambda_ = 0.001
                window_size = 6  
                d = np.sqrt(u_np[:, 0]**2 + u_np[:, 1]**2)
                deadlock_mask = d < 0.01
                if 'v_t' not in locals():
                    v_t = np.zeros_like(u_np)
                
                for i in range(args.num_agents):
                    if deadlock_mask[i]:
                        v_t[i] = np.zeros_like(u_np[i])
                        for j in range(1, 101):
                            if t - j >= 0:
                                v_t[i] += (gamma ** j) * u_values[t - j][i]
                        grad = np.gradient(u_np[i] - u_ref_np[i])**2
                        v_t[i] += lambda_ * grad
                        u_np[i, :] -= v_t[i]
                        dsdt_d = core.quadrotor_dynamics_np(s_np, u_np)
                        s_np[i, :] = s_np[i, :] + dsdt_d[i, :] * config.TIME_STEP_EVAL
                
                deadlock_ours.append(deadlock_mask)
                if len(deadlock_ours) > window_size:
                    deadlock_ours = deadlock_ours[-window_size:]
                # Calculate deadlock detection over the sliding window
                deadlock_window = np.array(deadlock_ours)
                deadlock_info_window = np.all(deadlock_window, axis=0).astype(np.float32)
        
                deadlock_info = np.sum(deadlock_info_window)
                #print("deadlock_info", deadlock_info)
             
                deadlock_ratio = deadlock_info / args.num_agents  
              
                deadlock_ratios_epoch.append(deadlock_ratio)  # Ensure values are <= 1
                deadlock_rate = np.sum(deadlock_ratios_epoch)
                leng = len(deadlock_ratios_epoch)
               
                
                if np.mean(
                    np.linalg.norm(s_np[:, :3] - s_ref_np[:, :3], axis=1)
                    ) < config.DIST_TOLERATE:
                    break
            
                s_traj.append(np.expand_dims(s_np[:, [0, 1, 2, 6, 7]], axis=0))
                current_step_index = 0
                collisions = core.dangerous_mask_np(s_np, config.DIST_MIN_CHECK)
                collision_tracking[current_step_index] = np.any(collisions, axis=1).astype(int)
                deadlock_tracking[current_step_index] = deadlock_mask.astype(int)  # Track deadlocks

                current_step_index += 1  # Move to the next global step
            u_values.append(u_np.copy())
        safety_reward.append(np.mean(safety_info))
        dist_reward.append(np.mean((np.linalg.norm(
            s_np[:, :3] - s_ref_np[:, :3], axis=1) < 1.5).astype(np.float32) * 10))
        dist_errors.append(
            np.mean(np.linalg.norm(s_np[:, :3] - s_ref_np[:, :3], axis=1)))
        


        def predict_collision(s_np, dsdt):
                s_np_next = s_np + dsdt * config.TIME_STEP_EVAL
                pairwise_diff = s_np_next[:, :3].reshape(-1, 1, 3) - s_np_next[:, :3].reshape(1, -1, 3)
                safety_distances = np.linalg.norm(pairwise_diff, axis=2)
                collision_mask = safety_distances < config.DIST_MIN_CHECK
                collision_prediction = collision_mask.astype(int)
                np.fill_diagonal(collision_prediction, 0)
                return collision_prediction
            
        def model_setup(s_np,s_ref, u_ref):
            """
            Setup the do_mpc model that describes the system dynamics, the cost function, and the barrier function.
            """
            model = do_mpc.model.Model('discrete')

            # Define states and control inputs
            _x = model.set_variable(var_type='_x', var_name='x', shape=(8, 1))
            _u = model.set_variable(var_type='_u', var_name='u', shape=(3, 1))

            # Dynamics function 
            A = config.A_MAT
            B = config.B_MAT
            dxdt = cs.mtimes(cs.DM(A), _x) + cs.mtimes(cs.DM(B), _u)
            s_np = s_np.reshape(8,1)
            x_next = s_np + dxdt * config.TIME_STEP
            model.set_rhs('x', x_next)


            # Cost function weights
            P = np.diag([100] * 8)
            Q = np.diag([1] * 3)
            s_ref_T = s_ref.reshape(8,1)
            u_ref_T = u_ref.reshape(3,1)
            cost_state = (_x - s_ref_T).T @ P @ (_x - s_ref_T)
            cost_control = (_u - u_ref_T).T @ Q @ (_u - u_ref_T)
            #cost_control = (_u).T @ Q @ (_u)
            model.set_expression(expr_name='stage_cost', expr=cost_state + cost_control)
            model.set_expression(expr_name='terminal_cost', expr=cost_state)

            model.setup()
            return model
        




        def setup_mpc(model,s_np_i, s_np):
            """
            Setup MPC controller using do_mpc library.

            """
            s_np_i = s_np_i.T 
            mpc = do_mpc.controller.MPC(model)
            setup_mpc = {
                'n_horizon': 1,
                't_step': config.TIME_STEP,
                'n_robust': 0,
                'store_full_solution': True,
            }
            mpc.set_param(**setup_mpc)

            mterm = model.aux['terminal_cost']  # Terminal cost
            lterm = model.aux['stage_cost']  # Stage cost

            mpc.set_objective(mterm=mterm, lterm=lterm)
            mpc.set_rterm(u=1e-2)  

            # Add constraints for each control input
            omega_x_limit = 0.2
            omega_y_limit = 0.2
            a_limit = 0.2
            max_u = np.array([omega_x_limit, omega_y_limit, a_limit])
            mpc.bounds['lower','_u', 'u'] = 0.1*max_u
            mpc.bounds['upper','_u', 'u'] = max_u

            s_np_new = s_np.reshape(8,1)[:3]
            
            # Safety barrier function
            def barrier_function(x, s_np_i, r=config.DIST_MIN_THRES):
                """
                x: The state of the current agent (shape [1, 8])
                s_np_i: States of other agents (shape [3, 8])
                r: Safety distance threshold
                """
                x = x[:3,:]
                x_rep = cs.repmat(x, 1, s_np_i.shape[1])
                diff = x_rep - s_np_i[:3, :]
                distances = cs.sqrt(cs.sum1(cs.power(diff, 2)))
                h = distances - r
                print("h", h.shape)
                return h
            
            def combined_cbf_constraint(h_values, kappa=10):
                exp_sum = cs.sum1(cs.exp(-kappa * cs.vertcat(*h_values)))
                combined_h = -cs.log(exp_sum)
                return combined_h
            

            h_x = barrier_function(model.x['x'], s_np_i, config.DIST_MIN_THRES)
            A = config.A_MAT
            B = config.B_MAT
            dxdt = cs.mtimes(cs.DM(A), model.x['x']) + cs.mtimes(cs.DM(B), model.u['u'])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            s_np_dxdt = s_np.reshape(8,1)
            x_next = s_np_dxdt + dxdt * config.TIME_STEP
            h_s_np = barrier_function(x_next, s_np_i, config.DIST_MIN_THRES)
            gamma = 0.8
            h_dot = (h_s_np - h_x)/config.TIME_STEP
            h_cons = -h_dot - gamma*h_x
            constraints = cs.horzsplit(h_cons)
            kappa = 50
            h_combined = combined_cbf_constraint(constraints, kappa)
            mpc.set_nl_cons('cbf_constraint_combined', h_combined, ub=0)
            mpc.setup()
            return mpc
        
        def run_mpc_control(mpc, model, initial_state, s_np_i):
            """
            Run MPC control step and retrieve optimized control actions.

            :param mpc: The MPC controller object.
            :param model: The MPC model object.
            :param initial_state: The initial state vector for the agent.
            :param s_np: The state array of all agents, used to update TVP settings.
            """
            model.x0 = initial_state
            mpc.x0 = initial_state

            mpc.set_initial_guess()
            u_opt = mpc.make_step(initial_state)
            print("u_opt", u_opt)

            return u_opt

        print("Predicting collision")
        dsdt = core.quadrotor_dynamics_np(s_np, u_np)  # Initial dynamics calculation
        iscollision = predict_collision(s_np, dsdt)
        print(iscollision)

        collision_resolved_count = 0
        mpc_icbf_trigger_count = 0  # Counter for MPC-ICBF triggers
        previous_iscollision = iscollision.copy()
        collision_pairs = set((i, j) for i in range(iscollision.shape[0]) for j in range(i + 1, iscollision.shape[1]) if iscollision[i, j] == 1)

        while collision_pairs:
            new_collision_pairs = set()
            
            for (i, j) in collision_pairs:
                print(f"Collision detected between agent {i} and agent {j}, updating controls.")
                
                # Update controls for agent i
                s_np_i = np.vstack([s_np[:i], s_np[i+1:]]) 
                model_i = model_setup(s_np[i,:], s_ref_np[i,:], u_ref_np[i,:])
                mpc_controller_i = setup_mpc(model_i, s_np_i, s_np[i, :])
                u_np[i, :] = run_mpc_control(mpc_controller_i, model_i, s_np[i, :], s_np_i).flatten()
                dsdt_i = core.quadrotor_dynamics_np(s_np[i,:], u_np[i,:])
                s_np[i,:] += dsdt_i * config.TIME_STEP

                # Update controls for agent j
                s_np_j = np.vstack([s_np[:j], s_np[j+1:]])
                model_j = model_setup(s_np[j,:], s_ref_np[j,:], u_ref_np[j,:])
                mpc_controller_j = setup_mpc(model_j, s_np_j, s_np[j, :])
                u_np[j, :] = run_mpc_control(mpc_controller_j, model_j, s_np[j, :], s_np_j).flatten()
                dsdt_j = core.quadrotor_dynamics_np(s_np[j,:], u_np[j,:])
                s_np[j,:] += dsdt_j * config.TIME_STEP

                print(f"Controls updated for agents {i} and {j} using mpc-cbf.")
                mpc_icbf_trigger_count += 1  # Increment the MPC-ICBF trigger counter
                

                dsdt = core.quadrotor_dynamics_np(s_np, u_np)
                iscollision = predict_collision(s_np, dsdt)
                new_collision_pairs = set((i, j) for i in range(iscollision.shape[0]) for j in range(i + 1, iscollision.shape[1]) if iscollision[i, j] == 1)
                
                if not np.array_equal(previous_iscollision, iscollision):
                    collision_resolved_count += 1
                    print(f"Collision resolved count: {collision_resolved_count}")
                    
                previous_iscollision = iscollision.copy()
                collision_pairs = new_collision_pairs

        print("All collisions resolved.")
        print(f"MPC-ICBF was triggered {mpc_icbf_trigger_count} times to resolve collisions.")


        dsdt = core.quadrotor_dynamics_np(s_np, u_np)
        print("Dynamics updated, now predicting collision again")
        
        for agent_idx in range(args.num_agents):
            steps_taken_by_agents[agent_idx, istep] = t + 1 

        traj_dict['ours'].append(np.concatenate(s_traj, axis=0))
        end_time = time.time()

        steps_taken_df = pd.DataFrame(steps_taken_by_agents, columns=['Evaluation Step {}'.format(i+1) for i in range(config.EVALUATE_STEPS)])
        steps_taken_df.to_csv('csv_data/steps_taken_by_agents.csv', index_label="Agent")
        print("Steps taken by agents saved to 'steps_taken_by_agents.csv'")

        # Visualization
        def initialize_trails(num_agents, num_trails, s_traj_shape, initial_positions):
            trails = np.tile(initial_positions[np.newaxis, :, :], (num_trails, 1, 1))
            return trails

        def update_trails(trails, current_positions):
            trails[:-1] = trails[1:]
            trails[-1] = current_positions 
            return trails

   
        num_trails = 16  
        alpha_values = np.linspace(0.1, 1.0, num_trails) 

        

        if args.vis > 0:

            s_traj_ours = traj_dict['ours'][-1]
            num_agents = s_traj_ours.shape[1]  
            initial_positions = s_traj_ours[0]
            trails = initialize_trails(num_agents, num_trails, s_traj_ours.shape, initial_positions)
            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin=0, vmax=num_trails)

            fig = plt.figure(figsize=(10, 10))
            ax_1 = fig.add_subplot(111, projection='3d')

            # Set the view limits to encompass the entire maze
            ax_1.set_xlim(0, 20)
            ax_1.set_ylim(0, 20)
            ax_1.set_zlim(0, 10)

            ax_1.axis('off')
            ax_1.grid(False)
            if args.env == 'Maze':
                show_obstacles(scene.OBSTACLES, ax_1) 
            ax_1.set_box_aspect([8, 8, 4])
            ax_1.view_init(elev=80, azim=-45)

            gif_frames = []

            for j in range(0, s_traj_ours.shape[0], 10):
                ax_1.clear()

                ax_1.set_xlim(0, 20)
                ax_1.set_ylim(0, 20)
                ax_1.set_zlim(0, 10)
                ax_1.axis('off')
                ax_1.grid(False)
                if args.env == 'Maze':
                    show_obstacles(scene.OBSTACLES, ax_1)
                j_ours = min(j // 10, len(deadlock_ours) - 1)
                s_np = s_traj_ours[min(j, s_traj_ours.shape[0]-1)]
                safety = safety_ours[min(j, s_traj_ours.shape[0]-1)]
                deadlock = deadlock_ours[j_ours]
                trails = update_trails(trails, s_np)
                current_trail_length = min((j // 10) + 1, num_trails)

                # Plot trailing points for each agent with varying color and opacity
                for t in range(current_trail_length):
                    color_value = cmap(norm(t))
                    ax_1.scatter(trails[t][:, 0], trails[t][:, 1], trails[t][:, 2],
                                color=color_value, alpha=alpha_values[t], s=100) 
                ax_1.scatter(s_np[deadlock, 0], s_np[deadlock, 1], s_np[deadlock, 2],
                            color='green', s=100, label='Deadlock')
                ax_1.scatter(scene.end_points[:, 0], scene.end_points[:, 1], scene.end_points[:, 2],
                            color='blue', s=100, label='Goal')
                ax_1.set_title('MA-ICBF: Safety Rate = {:.4f}, Deadlocked Agents = {:.4f}'.format(
                    np.mean(safety_ratios_epoch), np.mean(deadlock_info)), fontsize=22)

                plt.legend(loc='upper right', fontsize=22)
                fig.canvas.draw()
                plt.pause(0.001)
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                gif_frames.append(image)

            # Save the gif
            gif_path = os.path.join('trajectory', 'ours_trajectory_32_agents_empty_itr_099_fps_10_trailing_random.gif')
            imageio.mimsave(gif_path, gif_frames, fps=10)
            print(f"GIF saved at: {gif_path}")

        
        print('Evaluation Step: {} | {}, Time: {:.4f}, Deadlocked Agents: {:.4f}'.format(
            istep + 1, config.EVALUATE_STEPS, end_time - start_time, np.mean(deadlock_info)))
    

    collision_tracking = np.clip(collision_tracking, 0, 1)
    total_collisions = np.sum(collision_tracking)/2
    print('Total Number of Collisions :', total_collisions)
    base_directory = 'csv_data'
    sub_directory = 'collision_tracking'
    directory_path = os.path.join(base_directory, sub_directory)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    csv_file_path = os.path.join(directory_path, 'hu_time_baseline_4_agents_itr_03.csv')
    column_names = ['Agent {}'.format(i+1) for i in range(args.num_agents)]
    df_collision_tracking = pd.DataFrame(collision_tracking, columns=column_names)
    df_collision_tracking.to_csv(csv_file_path, index_label="Step")
    print("collision tracking data saved!!!")

    print_accuracy(accuracy_lists)
    print('Distance Error (MA-ICBF ): {:.4f}'.format(
          np.mean(dist_errors)))
    print('Mean Safety Rate (MA-ICBF ): {:.4f}'.format(
          np.mean(safety_ratios_epoch)))
    print('Deadlocked agents (MA-ICBF ): {:.4f}'.format(
          np.mean(deadlock_info)))  

    safety_reward = np.mean(safety_reward)
    dist_reward = np.mean(dist_reward)
    print('Safety Reward: {:.4f}, Dist Reward: {:.4f}, Reward: {:.4f}'.format(
        safety_reward, dist_reward, 9 + 0.1 * (safety_reward + dist_reward)))

    pickle.dump(traj_dict, open('trajectory/traj_eval.pkl', 'wb'))
    scene.write_trajectory('trajectory/env_traj_eval.pkl', traj_dict['ours'])



if __name__ == '__main__':
    main()





