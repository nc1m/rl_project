import sys
from gym import Env
from src import env

EPISODES = 200
EPISODE_MAX_STEP = 100


def main():
    
    agent = None
    enviroment = env.Enviroment('CartPole-v0')
    enviroment.play(EPISODES, EPISODE_MAX_STEP)


if __name__ == '__main__':
    sys.exit(main())  