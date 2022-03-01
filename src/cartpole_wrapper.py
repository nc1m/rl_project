import time
import gym
import numpy as np
from gym import spaces


def get_screen(env):
    ''' Extract one step of the simulation.'''
    screen = env.render(mode='rgb_array')
    screen = np.ascontiguousarray(screen, dtype=np.uint8)
    return screen


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(0, 255, (3, 400, 600), np.uint8)

    def observation(self, observation):
        return get_screen(self.env)


def main():
    from matplotlib import pyplot as plt
    env = gym.envs.make("CartPole-v1")
    env.reset()
    screen = env.render(mode='rgb_array')
    print(screen.shape)

    env = gym.envs.make("CartPole-v1")
    env.reset()

    env = ObservationWrapper(env)
    print(env.observation_space)

    for _ in range(10):
        env.render()
        obs, reward, _, _ = env.step(env.action_space.sample())
        plt.imshow(obs)
        plt.show()
        time.sleep(1)

    env.close()

if __name__ == '__main__':
    main()
