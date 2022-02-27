import sys
from gym import Env
from src import env
import argparse


ENVIROMENT_CHOICES = ['CartPole-v1', 'HumanoidStandup-v2']

def parse_args():
    parser = argparse.ArgumentParser(description="""
TODO Description""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-e', '--env',
                        default='CartPole-v0',
                        type=str,
                        choices=ENVIROMENT_CHOICES,
                        help='Set enviroment (see description above for details). Possible values are: '+', '.join(ENVIROMENT_CHOICES),
                        metavar='env')
    parser.add_argument('-s', '--seed', default=42, type=int,
                        help='Sets the seed for this experiment and consequently the random sample shown.')
    parser.add_argument('-eps', '--episodes', default=200, type=int,
                        help='The number of episode for the enviroment.')
    parser.add_argument('-mepss', '--maxEpisodeSteps', default=100, type=int,
                        help='The maximum episode steps in a episode')
    parser.add_argument('--no_cuda', action='store_true', help='Set this if cuda is availbable, but you do NOT want to use it.')
    parser.add_argument('--pobs', action='store_true', help='Prints the observation')

    return parser.parse_args()

def main(args):
    
    agent = None
    enviroment = env.Enviroment(args.env)
    enviroment.play(args.episodes, args.maxEpisodeSteps, agent, args.pobs)


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))  