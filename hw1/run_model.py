import tensorflow as tf
import numpy as np
import os
import json

from data_generator import DataGenerator
from model import Model
from trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args




def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # load data configs
    with open(os.path.join(config.summary_dir, "data_configs.json"), 'r') as f:
        data_configs = json.load(f)
    print(data_configs)

    for key, value in data_configs.items():
        config[key] = value

    # create tensorflow session
    sess = tf.Session()
    # create an instance of the model you want
    model = Model(config, is_training=False)
    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, config)
    #load model
    model.load(sess)
    
    # run environment
    import gym
    env = gym.make(config.envname)
    env.seed(0)
    max_steps = env.spec.timestep_limit
    returns = []
    observations = []
    actions = []

    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
        action = trainer.predict(np.expand_dims(obs,axis=0))[0]
        observations.append(obs)
        actions.append(action)
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if config.render:
            env.render()
        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
        if steps >= max_steps:
            break
    returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}


if __name__ == '__main__':
    main()