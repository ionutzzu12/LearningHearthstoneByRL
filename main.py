from training import *
import threading
import os
from time import sleep


max_episode_length = 300
gamma = .99  # discount rate for advantage estimation and reward discounting
s_size = 172  # depends on encoding version
a_size = 64
num_workers = 6  # set workers to number of available CPU threads, can be lowered on subsequent runs
num_testers = 2
load_model = False

# agent to train against
train_opponent = 'firstAction'
train_vs_self = train_opponent.lower() == 'self'

model_path = './simple_bot_play/T25-172-gamma99/model'
frames_path = './simple_bot_play/T25-172-gamma99/frames'

# best self play result
# model_path = './T13-172-zoo-6-SELF-gamma99-scale_reward/model'
# frames_path = './T13-172-zoo-6-SELF-gamma99-scale_reward/frames'


# just test performance against a specific opponent
just_competing = False
assert not just_competing or load_model
opponents = ['Random', 'FirstAction', 'MCTS', 'GSVB']  # TODO: MCTS and GSVB are too expensive


tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create a directory to save episode replays to
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

with tf.device("/gpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network


def get_new_worker(index):
    return Worker(index, s_size, a_size, trainer, model_path, frames_path, global_episodes)


workers = []
# Create worker classes
for i in range(num_workers):
    workers.append(get_new_worker(i))

saver = tf.train.Saver(max_to_keep=5)

# setup sess for gpu
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

coord = tf.train.Coordinator()

if load_model == True:
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())


# This is where the asynchronous magic happens.
# Start the "work" process for each worker in a separate threat.
worker_threads = []
i = 0

for worker in workers:
    first_tester_index = num_workers - num_testers

    if just_competing:
        worker_work = lambda: worker.test_values(opponents[i - first_tester_index], max_episode_length, gamma, sess, coord, saver)
        pass
    elif i >= first_tester_index:
        worker_work = lambda: worker.test(opponents[i - first_tester_index], max_episode_length, gamma, sess, coord, saver)
    elif train_vs_self:
        worker_work = lambda: worker.work_self_play(train_opponent, max_episode_length, gamma, sess, coord, saver)
    else:
        worker_work = lambda: worker.work(train_opponent, max_episode_length, gamma, sess, coord, saver)

    t = threading.Thread(target=(worker_work))
    t.start()
    sleep(0.5)
    worker_threads.append(t)
    i += 1

coord.join(worker_threads)
