# Programm env parameter name to gym env name
ENVIRONMENT_CHOICES = dict()
ENVIRONMENT_CHOICES['cart_pole'] = 'CartPole-v1'
ENVIRONMENT_CHOICES['pong'] = 'ALE/Pong-v5'
ENVIRONMENT_CHOICES['ms_pacman'] = 'ALE/MsPacman-v5'

# Default number of episodes
NUM_EPISODES = 50

# Default maximum number of steps in an episode
NUM_STEPS = 100

# Default replay memory size
REPLAY_MEMORY_SIZE = 10000

# Default batch size
BATCH_SIZE = 128
