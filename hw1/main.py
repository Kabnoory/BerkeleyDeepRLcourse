import os, sys
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

from data_generator import DataGenerator
from model import Model
from trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main(config):
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    # create an instance of the model you want
    model = Model(config, data.num_labels, data.dataset.output_types, data.dataset.output_shapes)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, config, logger)
    # load model if exists
    model.load(sess)
    # here you train your model
    trainer.fit(data)

def main_dagger(config):
    # create tensorflow session
    with tf.Session() as sess:
        # create the experiments dirs
        create_dirs([config.summary_dir, config.checkpoint_dir])
        # Load policy
        print('loading and building expert policy')
        policy_fn = load_policy.load_policy(config["dagger"]["expert_policy_file"])
        print('loaded and built')
        # setup gym environment
        import gym
        env = gym.make(config.envname)
        max_steps = env.spec.timestep_limit

        observations = []
        actions = []
        for j in range(config["dagger"]["dagger_iter"]):
            print("DAgger iteration #%d..."%(j+1))
            for i in range(config["dagger"]["num_rollouts"]): 
                obs = env.reset()
                done = False
                steps = 0
                while not done:
                    correct_action = policy_fn(np.expand_dims(obs,axis=0))
                    action = correct_action if (j == 0) else trainer.predict(np.expand_dims(obs,axis=0))[0]
                    observations.append(obs)
                    actions.append(correct_action)
                    obs, _, done, _ = env.step(action)
                    steps += 1
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break

                expert_data = {'observations': np.array(observations),
                               'actions': np.array(actions)}

                s = 'wb' if j == 0 else 'ab' # append to file after first iteration
                with open(os.path.join('..', '..', 'expert_data', config.envname + '_DAgger.pkl'), s) as f:
                    pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

            if j == 0: # setup model architecture
                # create your data generator
                data = DataGenerator(config, raw=expert_data)
                # create an instance of the model you want
                model = Model(config, data.num_labels, data.dataset.output_types, data.dataset.output_shapes)
                # create tensorboard logger
                logger = Logger(sess, config)
                # create trainer and pass all the previous components to it
                trainer = Trainer(sess, model, config, logger)
                # here you train your model
                trainer.fit(data)
            else:
                data.load_raw_data(expert_data)
                trainer.fit(data)


if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    if "dagger" in config:
        main_dagger(config) # perform dagger algorithm
    else:
        main(config)
