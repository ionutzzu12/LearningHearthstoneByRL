"""
Credits: https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import time

from spellsource_env import *


# ### Helper Functions


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):  # TODO: initialized with ones, previously was random
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.ones(shape=shape, dtype=np.float64)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


# ### Actor-Critic Network


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float64)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, s_size, 1])

            # convolutions
            '''
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.imageIn, num_outputs=64,
                                     kernel_size=[1, 3], stride=[1, 3], padding='VALID')
            '''
            self.hidden_1 = slim.fully_connected(slim.flatten(self.imageIn), 256, activation_fn=tf.nn.elu)
            self.hidden_2 = slim.fully_connected(self.hidden_1, 128, activation_fn=tf.nn.elu)
            # last_hidden = slim.fully_connected(self.hidden_2, 64, activation_fn=tf.nn.elu)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float64)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float64)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float64, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float64, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(self.hidden_2, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int64)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float64)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float64)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float64)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


# ### Worker Agent


class Worker:
    def __init__(self, name, s_size, a_size, trainer, model_path, frames_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.frames_path = frames_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(model_path + "/train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.env = SpellsourceEnv.init()
        # self.actions = SpellsourceEnv.get_num_actions()

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        # next_observations = rollout[:, 3]
        values = rollout[:, 5]
        final_reward = rewards[-1]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # TODO: is this ZERO-SUM?
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)
        """
        print("discounted_rewards", discounted_rewards)
        print("discounted_advntgs", advantages)
        input("Press Enter to continue...")
        
        discounted_rewards = rewards
        advantages = discounted_rewards + values[1:] - values[:-1]
        """
        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,
                                                                     self.local_AC.policy_loss,
                                                                     self.local_AC.entropy,
                                                                     self.local_AC.grad_norms,
                                                                     self.local_AC.var_norms,
                                                                     self.local_AC.state_out,
                                                                     self.local_AC.apply_grads],
                                                                    feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, opponent, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        start_count = episode_count
        print("[TRAINING] Starting worker_" + str(self.number), "episode:", episode_count, "opponent:", opponent)
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0

                save_current_episode = episode_count % 500 == 0 and self.name == 'worker_0' \
                                       and not start_count == episode_count

                s, r, d, va, p = self.env.reset(opponent)
                assert p == 1
                done = d

                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state

                while not done:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: [s],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})

                    num_valid_actions = len(va)
                    valid_probs = a_dist[0][:num_valid_actions]

                    if save_current_episode:  # play as best as it can for the audience
                        a = np.argmax(valid_probs)
                    else:  # add some exploration
                        valid_probs = valid_probs / valid_probs.sum()  # normalize
                        a = np.random.choice(valid_probs, p=valid_probs)
                        a = np.argmax(valid_probs == a)
                    a = a.item()

                    s1, r, d, va, p = self.env.step(a)
                    assert p == 1
                    done = d

                    if done:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    episode_step_count += 1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save replays of episodes, model parameters, and summary statistics.
                if save_current_episode:
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    print("Saved Model")
                    try:
                        self.env.save_replay(self.frames_path + '/match-' + str(episode_count) + '.txt')
                    except:
                        print("Something went wrong with the replay")

                display_every_n = 100

                if episode_count % display_every_n == 0 and not start_count == episode_count:
                    mean_reward = np.mean(self.episode_rewards[-display_every_n:])
                    mean_length = np.mean(self.episode_lengths[-display_every_n:])
                    mean_value = np.mean(self.episode_mean_values[-display_every_n:])
                    print(self.name, episode_count, mean_reward, mean_length, mean_value)

                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                episode_count += 1
                if self.name == 'worker_0':
                    sess.run(self.increment)

    def work_self_play(self, train_opponent, max_episode_length, gamma, sess, coord, saver):
        assert train_opponent.lower() == 'self'
        episode_count = sess.run(self.global_episodes)
        start_count = episode_count
        display_every_n = 20
        pos_episode_rewards = []
        neg_episode_rewards = []
        print("[SELF PLAY] Starting worker_" + str(self.number), "episode", episode_count)

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = [[], []]
                episode_values = [[], []]
                episode_reward = 0
                episode_step_count = 0  # number of turns

                save_current_episode = episode_count % 20 == 0 and self.name == 'worker_0' \
                                       and not start_count == episode_count

                s, r, d, va, current_player = self.env.reset('self')
                done = d

                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state

                while not done:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: [s],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})

                    num_valid_actions = len(va)
                    valid_probs = a_dist[0][:num_valid_actions]

                    if save_current_episode:  # play as best as it can
                        a = np.argmax(valid_probs)
                    else:  # ++exploration
                        valid_probs = valid_probs / valid_probs.sum()  # normalize
                        a = np.random.choice(valid_probs, p=valid_probs)
                        a = np.argmax(valid_probs == a)
                    a = a.item()

                    # ####### DEBUG ########
                    """
                    print('Player', current_player)
                    print('State', s)
                    print('Action', va[a])
                    input("Press Enter to continue...")
                    """
                    # ##### END DEBUG ######

                    s1, r, d, va, next_player = self.env.step(a)
                    done = d

                    if done:  # somebody won, hopefully
                        if r == 1:
                            winner = 1
                        elif r == -1:
                            winner = 0
                        else:
                            winner = -1
                            # print('TIE', episode_count, r)

                        # TODO: keep the best version
                        loser = 1 - winner
                        # reward_value = 1
                        # punishment_value = -1

                        # scale reward
                        reward_value = 1
                        punishment_value = -1
                        # reward_value = 1
                        # punishment_value = 0  # set to zero

                        if winner == -1:
                            punishment_value = reward_value = -0.5  # punish tie
                            winner = 1
                            loser = 0

                        if winner == current_player:  # player 0 won
                            episode_buffer[winner].append([s, a, reward_value, s, d, v[0, 0]])
                            episode_values[winner].append(v[0, 0])
                            episode_buffer[loser][-1][2] = punishment_value
                        else:
                            episode_buffer[loser].append([s, a, punishment_value, s, d, v[0, 0]])
                            episode_values[loser].append(v[0, 0])
                            episode_buffer[winner][-1][2] = reward_value

                        assert episode_buffer[winner][-1][2] == reward_value
                        assert episode_buffer[loser][-1][2] == punishment_value

                        pos_episode_rewards.append(r)
                        neg_episode_rewards.append(r / episode_step_count)
                        # ####### DEBUG ########
                        """
                        if episode_count % display_every_n == 0 and not start_count == episode_count:
                            print_trace(self.env.my_beh.context)
                            print("winner", winner, "turns", episode_step_count)
                            print("player 0", np.array(episode_buffer[0])[:, 2])
                            print("player 1", np.array(episode_buffer[1])[:, 2])
                        """
                        # ##### END DEBUG ######
                    else:
                        assert r == 0
                        if current_player == next_player:
                            episode_buffer[current_player].append([s, a, r, s1, d, v[0, 0]])
                            episode_values[current_player].append(v[0, 0])
                        else:
                            episode_buffer[current_player].append([s, a, r, s, d, v[0, 0]])  # ended turn
                            episode_values[current_player].append(v[0, 0])
                            episode_step_count += 1
                    # TODO: maybe append just the end turn state?
                    s = s1
                    current_player = next_player

                # self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values[0]))

                # Update the network using the episode buffer at the end of the episode.
                v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer[0], sess, gamma, 0.0)
                v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer[1], sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if save_current_episode:
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    print("Saved Model")
                    try:
                        self.env.save_replay(self.frames_path + '/match-' + str(episode_count) + '.txt')
                    except:
                        print("Something went wrong with the replay")

                if episode_count % display_every_n == 0 and not start_count == episode_count:
                    mean_reward_p = np.mean(pos_episode_rewards[-display_every_n:])
                    mean_reward_n = np.mean(neg_episode_rewards[-display_every_n:])
                    mean_length = np.mean(self.episode_lengths[-display_every_n:])
                    mean_value = np.mean(self.episode_mean_values[-display_every_n:])

                    print(self.name, episode_count, mean_reward_p, mean_length, mean_value)

                    summary = tf.Summary()
                    summary.value.add(tag='Perf/PositiveReward', simple_value=float(mean_reward_p))
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward_n))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                episode_count += 1
                if self.name == 'worker_0':
                    sess.run(self.increment)

    def test(self, opponent, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        start_count = episode_count
        print("[POLICY TEST] Starting worker_" + str(self.number), "episode", start_count, ":: testing win rate versus",
              opponent)
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                episode_count += 1

                s, r, d, va, player = self.env.reset(opponent=opponent, disable_fatigue=False)
                assert player == 1
                done = d

                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state

                while not done:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: [s],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})

                    num_valid_actions = len(va)
                    valid_probs = a_dist[0][:num_valid_actions]

                    a = np.argmax(valid_probs)
                    a = a.item()

                    s1, r, d, va, player = self.env.step(a)
                    assert player == 1
                    done = d

                    episode_values.append(v[0, 0])
                    episode_reward += r
                    s = s1
                    episode_step_count += 1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                display_every_n = 50

                if episode_count % display_every_n == 0 and not start_count == episode_count:
                    episode_count = sess.run(self.global_episodes)
                    episode_count = episode_count - episode_count % display_every_n

                    mean_reward = np.mean(self.episode_rewards[-display_every_n:])
                    mean_length = np.mean(self.episode_lengths[-display_every_n:])
                    mean_value = np.mean(self.episode_mean_values[-display_every_n:])
                    print(self.name, episode_count, mean_reward, mean_length, mean_value)

                    summary = tf.Summary()
                    summary.value.add(tag='Test/WinRate', simple_value=float((mean_reward + 1) / 2))
                    summary.value.add(tag='Test/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Test/Value', simple_value=float(mean_value))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                    try:
                        self.env.save_replay(self.frames_path + '/match-' + str(opponent) + str(episode_count) + '.txt')
                    except:
                        print("Something went wrong with the replay")

                    print(self.name, episode_count, ':: tester going to sleep for 10 mins...')
                    time.sleep(600)
                    print(self.name, episode_count, 'tester woke up!')
                    sess.run(self.update_local_ops)

    # TODO: the server does not support this right now
    """
    def test_values(self, opponent, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        start_count = episode_count
        wrong_shape_count = 0
        print("[VALUES TEST] Starting worker_" + str(self.number), "episode", start_count, ":: testing win rate versus", opponent)
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                episode_count += 1

                s, r, d, va, player = self.env.reset(opponent)
                context = self.env.my_beh.context
                # context.set_precompute_rollouts()
                context.disable_fatigue()

                assert player == 1
                done = d

                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state

                while not done:
                    # Take an action using VALUES
                    encoded_states = []

                    for action in va:
                        simulation = context.clone()
                        print_action(action, context)
                        print_action(action, simulation)
                        simulation.getLogic().performGameAction(player, action)
                        encoded_state = np.array(list(simulation.java_encode_state_v3_updated_with_many_stats()))
                        encoded_states.append(encoded_state)

                    a_dist, values, rnn_state = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs: encoded_states,
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})

                    num_valid_actions = len(va)
                    print(values.shape, num_valid_actions)
                    print(values)

                    if not num_valid_actions == values.shape[0]:
                        wrong_shape_count += 1
                    a = np.argmax(values[:num_valid_actions])
                    a = a.item()

                    s1, r, d, va, player = self.env.step(a)
                    assert player == 1
                    done = d

                    episode_values.append(values[a])
                    episode_reward += r
                    s = s1
                    episode_step_count += 1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                display_every_n = 1

                if episode_count % display_every_n == 0 and not start_count == episode_count:
                    episode_count = sess.run(self.global_episodes)
                    episode_count = episode_count - episode_count % 100

                    mean_reward = np.mean(self.episode_rewards[-display_every_n:])
                    mean_length = np.mean(self.episode_lengths[-display_every_n:])
                    mean_value = np.mean(self.episode_mean_values[-display_every_n:])
                    print(self.name, episode_count, mean_reward, mean_length, mean_value, wrong_shape_count)

                    summary = tf.Summary()
                    summary.value.add(tag='TestValues/WinRate', simple_value=float((mean_reward + 1) / 2))
                    summary.value.add(tag='TestValues/Length', simple_value=float(mean_length))
                    summary.value.add(tag='TestValues/Value', simple_value=float(mean_value))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                    self.env.save_replay(self.frames_path + '/test_values-' + opponent + str(episode_count) + '.txt')
    """
