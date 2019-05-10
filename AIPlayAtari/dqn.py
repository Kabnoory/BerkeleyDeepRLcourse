import uuid
import time
import pickle
import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class QLearner(object):

  def __init__(
    self,
    env,
    q_func,
    optimizer_spec,
    session,
    exploration=LinearSchedule(1000000, 0.1),
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=10,
    rew_file=None,
    double_q=True,
    lander=False):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    self.target_update_freq = target_update_freq
    self.optimizer_spec = optimizer_spec
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.stopping_criterion = stopping_criterion
    self.env = env
    self.session = session
    self.exploration = exploration
    self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file

    ###############
    # BUILD MODEL #
    ###############

    if len(self.env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = self.env.observation_space.shape
    else:
        img_h, img_w, img_c = self.env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    self.num_actions = self.env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    self.obs_t_ph              = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    self.act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    self.rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    self.obs_tp1_ph            = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    self.done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    if lander:
      obs_t_float = self.obs_t_ph
      obs_tp1_float = self.obs_tp1_ph
    else:
      obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
      obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

    # build Q network
    self.model_out = q_func(obs_t_float, self.num_actions, scope="q_func", reuse=False)
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    
    # build target network
    target_q = q_func(obs_tp1_float, self.num_actions, scope="target_q_func", reuse=False)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    # compute the Bellman error
    max_target_q = tf.reduce_max(target_q, axis=1)
    target = self.rew_t_ph + (1 - self.done_mask_ph) * gamma * max_target_q
    pred = tf.reduce_sum(tf.one_hot(self.act_t_ph, self.num_actions) * self.model_out, axis=1)
    self.total_error = tf.reduce_mean(huber_loss(pred - target))

    # construct optimization op (with gradient clipping)
    self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
    self.train_fn = minimize_and_clip(optimizer, self.total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    self.update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
    self.replay_buffer_idx = None

    ###############
    # RUN ENV     #
    ###############
    self.model_initialized = False
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    self.last_obs = self.env.reset()
    self.log_every_n_steps = 10000

    self.start_time = None
    self.t = 0

  def stopping_criterion_met(self):
    return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

  def step_env(self):
    ### 2. Step the env and store the transition

    # add new frame
    frame_id = self.replay_buffer.store_frame(self.last_obs)
    frame_obs = self.replay_buffer.encode_recent_observation()

    # get best next action or epsilon greedy exploration
    if np.random.random() < self.exploration.value(self.t) or not self.model_initialized:
      if self.model_initialized:
        print("WORKS")
      # action_space.sample() gives a random action
      action = self.env.action_space.sample()
      # print("RANDOM ACTION: ", action)
    else:
      best_action = tf.argmax(self.model_out,axis=1)
      action = self.session.run(best_action, feed_dict={self.obs_t_ph:[frame_obs]})[0]
      # print("ACTION: ", action)

    obs, reward, done, info = self.env.step(action)
    self.replay_buffer.store_effect(frame_id, action, reward, done)

    if done:
      self.last_obs = self.env.reset()
    else:
      self.last_obs = obs

  def update_model(self):
    ### 3. Perform experience replay and train the network.

    # note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):

      # sample a batch of transitions
      obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = \
        self.replay_buffer.sample(self.batch_size)

      # initalize model if it has not been initialized yet
      if not self.model_initialized:
        initialize_interdependent_variables(self.session, tf.global_variables(), {
          self.obs_t_ph: obs_batch,
          self.obs_tp1_ph: next_obs_batch
        })
        self.model_initialized = True

      # train the model
      self.session.run(self.train_fn, 
        feed_dict = {
          self.obs_t_ph: obs_batch,
          self.act_t_ph: act_batch,
          self.rew_t_ph: rew_batch,
          self.obs_tp1_ph: next_obs_batch,
          self.done_mask_ph: done_mask,
          self.learning_rate: self.optimizer_spec.lr_schedule.value(self.t)
        })

      # periodically update the target network
      if self.num_param_updates % self.target_update_freq == 0:
        self.session.run(self.update_target_fn)
      self.num_param_updates += 1

    self.t += 1

  def log_progress(self):
    episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

    if len(episode_rewards) > 0:
      self.mean_episode_reward = np.mean(episode_rewards[-100:])

    if len(episode_rewards) > 100:
      self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

    if self.t % self.log_every_n_steps == 0 and self.model_initialized:
      print("Timestep %d" % (self.t,))
      print("mean reward (100 episodes) %f" % self.mean_episode_reward)
      print("best mean reward %f" % self.best_mean_episode_reward)
      print("episodes %d" % len(episode_rewards))
      print("exploration %f" % self.exploration.value(self.t))
      print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
      if self.start_time is not None:
        print("running time %f" % ((time.time() - self.start_time) / 60.))

      self.start_time = time.time()

      sys.stdout.flush()

      with open(self.rew_file, 'wb') as f:
        pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)

def learn(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  while not alg.stopping_criterion_met():
    alg.step_env()
    # kwargs["env"].render()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
    alg.update_model()
    alg.log_progress()

