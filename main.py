import config
import core
import tensorflow as tf
import numpy as np
import argparse
import time
import os
import sys
import wandb
import csv
sys.dont_write_bytecode = True

np.set_printoptions(3)


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--num_agents', type=lambda x: [int(a) for a in x.split(',')], default='16')
    parser.add_argument('--num_agents', type=int, default=4)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tag', type=str, default='default')
    args = parser.parse_args()
    return args


def build_optimizer(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE)
    trainable_vars = tf.trainable_variables()

    # tensor to accumulate gradients over multiple steps
    accumulators = [
        tf.Variable(
            tf.zeros_like(tv.initialized_value()),
            trainable=False
        ) for tv in trainable_vars]
    # count how many steps we have accumulated
    accumulation_counter = tf.Variable(0.0, trainable=False)
    grad_pairs = optimizer.compute_gradients(loss, trainable_vars)
    # add the gradient to the accumulation tensor
    accumulate_ops = [
        accumulator.assign_add(
            grad
        ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]

    accumulate_ops.append(accumulation_counter.assign_add(1.0))
    # divide the accumulated gradient by the number of accumulation steps
    gradient_vars = [(accumulator / accumulation_counter, var)
                     for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]
    # separate the gradient of CBF and the controller
    gradient_vars_h = []
    gradient_vars_a = []
    for accumulate_grad, var in gradient_vars:
        if 'cbf' in var.name:
            gradient_vars_h.append((accumulate_grad, var))
        elif 'action' in var.name:
            gradient_vars_a.append((accumulate_grad, var))
        else:
            raise ValueError

    train_step_h = optimizer.apply_gradients(gradient_vars_h)
    train_step_a = optimizer.apply_gradients(gradient_vars_a)
    # re-initialize the accumulation tensor and accumulation step to zero
    zero_ops = [
        accumulator.assign(
            tf.zeros_like(tv)
        ) for (accumulator, tv) in zip(accumulators, trainable_vars)]
    zero_ops.append(accumulation_counter.assign(0.0))

    return zero_ops, accumulate_ops, train_step_h, train_step_a


def build_training_graph(num_agents, weight_loss_safe_dang, weight_loss_agile):
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
    
    # compute the value of loss functions and the accuracies
    loss_dang, loss_safe, acc_dang, acc_safe = core.loss_barrier(
        h=h, s=s, indices=indices)
    loss_dang_ic, loss_safe_ic, acc_dang_ic, acc_safe_ic = core.loss_barrier_ic(
        u=u, indices=indices)
    loss_agile = core.loss_agile(s=s, s_ref=s_ref, u=u, v_max=0.2, sigma_tight=0.05)

    (loss_dang_deriv, loss_safe_deriv, loss_medium_deriv, acc_dang_deriv,
     acc_safe_deriv, acc_medium_deriv) = core.loss_derivatives(
        s=s, u=u, h=h, x=x, indices=indices)
    loss_action = core.loss_actions(s=s, u=u, s_ref=s_ref, indices=indices)

    # Adjust the weights in the loss_list as per the parameters
    loss_list = [loss_dang * weight_loss_safe_dang, loss_safe * weight_loss_safe_dang, 3 * loss_dang_deriv,
                 loss_safe_deriv, 2 * loss_medium_deriv, 0.5 * loss_action, 
                 loss_dang_ic * weight_loss_safe_dang, loss_safe_ic * weight_loss_safe_dang, 
                 loss_agile * weight_loss_agile]
    acc_list = [acc_dang, acc_safe, acc_dang_deriv,
                acc_safe_deriv, acc_medium_deriv, acc_dang_ic, acc_safe_ic]

    weight_loss = [
        config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()]
    loss = 10 * tf.math.add_n(loss_list + weight_loss)

    return s, s_ref, u, loss_list, loss, acc_list


def count_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_list.append(np.mean(acc[acc[:, i] >= 0, i]))
    return acc_list


def main():
    args = parse_args()

    base_num_agents = args.num_agents
    num_agents_list = [base_num_agents, 2*base_num_agents, 4*base_num_agents, 8*base_num_agents]

    weights_safe_dang = [0.5, 1.0, 2.0, 5.0]
    weights_agile = [0.1, 0.5, 1.0, 2.0]
    
    for num_agents in num_agents_list:
        for weight_safe_dang in weights_safe_dang:
            for weight_agile in weights_agile:
                dynamic_name = f"h(u)_{weight_safe_dang}_{weight_agile}_{num_agents}"
                wandb.init(project="maicbf", name=dynamic_name, reinit=True, config={
                "num_agents": num_agents, 
                "weight_safe_dang": weight_safe_dang, 
                "weight_agile": weight_agile,
                "gpu": args.gpu,
                # Include any other arguments that need to be part of the config
            })
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

                tf.reset_default_graph()
                s, s_ref, u, loss_list, loss, acc_list = build_training_graph(num_agents, weight_safe_dang, weight_agile)
                zero_ops, accumulate_ops, train_step_h, train_step_a = build_optimizer(loss)

                accumulate_ops.append(loss_list)
                accumulate_ops.append(acc_list)

                accumulation_steps = config.INNER_LOOPS

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    saver = tf.train.Saver()

                    log_file_path = f'train_logs/training_log_{dynamic_name}.txt'
                    csv_file_path = f'csv_data/losses/losses_{dynamic_name}.csv'
                    model_path = f'models/model_{dynamic_name}'

                    with open(log_file_path, 'w') as log_file, open(csv_file_path, 'w', newline='') as csvfile:
                        loss_writer = csv.writer(csvfile)
                        loss_writer.writerow(['Step', 'Loss', 'Accuracy', 'Dist Error', 'Safety Ratio', 's', 's_ref', 'u'])  

                        if args.model_path:
                            saver.restore(sess, args.model_path)

                        loss_lists_np = []
                        acc_lists_np = []
                        dist_errors_np = []
                        dist_errors_baseline_np = []

                        safety_ratios_epoch = []
                        safety_ratios_epoch_baseline = []
                    
                        scene = core.Cityscape(args.num_agents)
                        start_time = time.time()
                        
                        for istep in range(config.TRAIN_STEPS):
                            scene.reset()
                            s_np = np.concatenate(
                                [scene.start_points, np.zeros((args.num_agents, 5))], axis=1)
                            sess.run(zero_ops)
                            for t in range(scene.steps):
                                s_ref_np = np.concatenate(
                                    [scene.waypoints[t], np.zeros((args.num_agents, 5))], axis=1)
                                for i in range(accumulation_steps):
                                    u_np, out = sess.run([u, accumulate_ops], feed_dict={
                                                        s: s_np, s_ref: s_ref_np})
                                    dsdt = core.quadrotor_dynamics_np(s_np, u_np)
                                    s_np = s_np + dsdt * config.TIME_STEP
                                    safety_ratio = 1 - np.mean(
                                        core.dangerous_mask_np(s_np, config.DIST_MIN_CHECK), axis=1)
                                    safety_ratio = np.mean(safety_ratio == 1)
                                    safety_ratios_epoch.append(safety_ratio)
                                    loss_list_np, acc_list_np = out[-2], out[-1]
                                    loss_lists_np.append(loss_list_np)
                                    acc_lists_np.append(acc_list_np)

                            dist_errors_np.append(np.mean(np.linalg.norm(
                                s_np[:, :3] - s_ref_np[:, :3], axis=1)))

                            if np.mod(istep // 10, 2) == 0:
                                sess.run(train_step_h)
                            else:
                                sess.run(train_step_a)

                            if np.mod(istep, config.DISPLAY_STEPS) == 0:
                                print('Step: {}, Time: {:.1f}, Loss: {}, Dist: {:.3f}, Safety Rate: {:.3f}'.format(
                                    istep, time.time() - start_time, np.mean(loss_lists_np, axis=0),
                                    np.mean(dist_errors_np), np.mean(safety_ratios_epoch)))
                                start_time = time.time()

                                
                            # Log training metrics to wandb
                                wandb.log({"Loss_agile": float(np.mean(loss_lists_np[8], axis=0)),
                                        "Loss_safe_ic": float(np.mean(loss_lists_np[7], axis=0)),
                                        "Loss_dang_ic": float(np.mean(loss_lists_np[6], axis=0)),
                                        "Accuracy": float(np.mean(acc_lists_np[1])),
                                        "Dist Error": float(np.mean(dist_errors_np)),
                                        "Safety Ratio": float(np.mean(safety_ratios_epoch)),
                                        "s": float(s_np[0, 0]),
                                        "s_ref": float(s_ref_np[0, 0]),
                                        "u": float(u_np[0, 0]),
                                        "Step": istep})

                                # Write training progress to text file and save losses to CSV within the with block
                                log_message = f'Step: {istep}, Time: {time.time() - start_time:.1f}, Loss: {np.mean(loss_lists_np, axis=0)}, Dist: {np.mean(dist_errors_np):.3f}, Safety Rate: {np.mean(safety_ratios_epoch):.3f}\n'
                                log_file.write(log_message)
                                loss_writer.writerow([istep, np.mean(loss_lists_np, axis=0), np.mean(acc_lists_np), np.mean(dist_errors_np), np.mean(safety_ratios_epoch), s_np[0, 0], s_ref_np[0, 0], u_np[0, 0]])

                                # print(np.mean(loss_lists_np))
                                # print(loss_lists_np)
                                (loss_lists_np, acc_lists_np, dist_errors_np, dist_errors_baseline_np, safety_ratios_epoch,
                                safety_ratios_epoch_baseline) = [], [], [], [], [], []
                            

                            if np.mod(istep, config.SAVE_STEPS) == 0 or istep + 1 == config.TRAIN_STEPS:
                                saver.save(sess, f'{model_path}_iter_{istep}')
                wandb.finish()
if __name__ == '__main__':
    main()
