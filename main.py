import argparse
import os
import random
import time
import logging
from datetime import timedelta

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import gym

from src.cartpole_wrapper import ObservationWrapper
from src.utils import ReplayMemory
from src.utils import Transition
from src.config import ENVIRONMENT_CHOICES
from src.config import NUM_EPISODES
from src.config import NUM_STEPS
from src.config import REPLAY_MEMORY_SIZE
from src.config import BATCH_SIZE


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', default=42, type=int,
                        help='Sets the seed for this experiment and consequently the random sample shown.')
    parser.add_argument('--no_cuda', action='store_true', help='Set this if cuda is availbable, but you do NOT want to use it.')
    parser.add_argument('--vis', action='store_true', help='visualize env')
    parser.add_argument('-e', '--env',
                        default='cart_pole',
                        type=str,
                        choices=[*list(ENVIRONMENT_CHOICES.keys())],
                        help='Set environment (see description above for details). Possible values are: '+', '.join(list(ENVIRONMENT_CHOICES.keys())),
                        metavar='ENV')
    parser.add_argument('-a', action='store_true')
    parser.add_argument('--replayMem', default=REPLAY_MEMORY_SIZE, type=int, help='Size of replay memory.')
    parser.add_argument('--numEp', default=NUM_EPISODES, type=int, help='Number of episodes')
    parser.add_argument('--numSteps', default=NUM_STEPS, type=int, help='Maximum number of steps in an epsiode')
    parser.add_argument('--batchSize', default=BATCH_SIZE, type=int, help='Batch size')
    parser.add_argument('--pobs', action='store_true', help='Prints the observation')
    return parser.parse_args()


def p_env_info(env):
    print(f'env.action_space: {env.action_space}')
    print(f'env.observation_space: {env.observation_space}')


def optimize_model(replayBuffer, batchSize):
    if len(replayBuffer) < batchSize:
        return

    transitions = replayBuffer.sample(batchSize)
    return
    # print(f'transitions.shape: {transitions.shape}')
    print(f'batchSize: {batchSize}')
    print(f'type(transitions): {type(transitions)}')
    print(f'type(transitions[0]): {type(transitions[0])}')
    print(f'type(transitions[0][0]): {type(transitions[0][0])}')
    print(f'transitions[0][0].shape: {transitions[0][0].shape}')
    batch = Transition(*zip(*transitions))  # From list(Transition) => Transition.Tupels
    # print(f'batch: {batch}')
    print(f'type(batch): {type(batch)}')
    print(f'type(batch[0]): {type(batch[0])}')
    print(f'len(batch[0]): {len(batch[0])}')
    print(f'type(batch[0][0]): {type(batch[0][0])}')
    print(f'batch[0][0].shape: {batch[0][0].shape}')


def main(args):
    set_seed(args.seed)
    startTime = time.time()
    use_cuda = False
    if torch.cuda.is_available() and not args.no_cuda:
        # pass # TODO
        use_cuda = True
    print('CUDA enabled:', use_cuda)

    print(f'args.env: {args.env}')
    envName = ENVIRONMENT_CHOICES[args.env]
    print(f'envName: {envName}')
    use_ale = False
    if envName.startswith('ALE/'):
        use_ale = True
    print(f'use_ale: {use_ale}')

    if args.vis:
        if use_ale:
            env = gym.make(envName, render_mode='human')
        else:
            env = gym.make(envName)
    else:
        env = gym.make(envName)

    env.seed(args.seed)

    if args.env == 'cart_pole':
        env = ObservationWrapper(env)

    p_env_info(env)

    replayBuffer = ReplayMemory(args.replayMem)

    for i_episode in range(args.numEp):
        curState = env.reset()
        for i_steps in range(args.numSteps):
            if not use_ale and args.vis:
                env.render()
            # time.sleep(0.1)
            # print(observation)
            action = env.action_space.sample()
            nextState, reward, done, info = env.step(action)
            # print(f'nextState.min(): {nextState.min()}')
            # print(f'nextState.max(): {nextState.max()}')
            if i_episode == 0 and i_steps == 0:
                print(f'action: {action}')
                print(f'type(action): {type(action)}')
                print(f'curState.shape: {curState.shape}')
                print(f'curState.dtype: {curState.dtype}')
                print(f'type(curState): {type(curState)}')
                print(f'reward: {reward}')
                print(f'type(reward): {type(reward)}')
                print(f'done: {done}')
                print(f'type(done): {type(done)}')
                print(f'info: {info}')
                print(f'type(info): {type(info)}')
            replayBuffer.push(curState, action, nextState, reward)
            curState = nextState

            optimize_model(replayBuffer, args.batchSize)
            if done:
                print("Episode finished after {} timesteps".format(i_steps+1))
                break
    env.close()

    # OLD
    # print(gym.envs.registry.all())
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=T.InterpolationMode.BICUBIC),#Image.CUBIC),
                        T.ToTensor()])

    # agent = None
    # enviroment = env.Enviroment(args.env)
    # enviroment.play(args.episodes, args.maxEpisodeSteps, agent, args.pobs)

    print(f'run time: {str(timedelta(seconds=(time.time() - startTime)))}')


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if not args.a:
        main(args)
    else:
        for envKey, envName in ENVIRONMENT_CHOICES.items():
            print(f'###############################{envName}################################')
            args.env = envKey
            main(args)
            print('\n\n')
