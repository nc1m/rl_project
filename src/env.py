import gym 

class Enviroment():
    def __init__(self, env_str):
        self.env = gym.make(env_str)
        
    def play(self, EPISODES, EPISODE_MAX_STEP, agent=None, pobs=False):
        for i_episode in range(EPISODES):
            observation = self.env.reset()
            for t in range(EPISODE_MAX_STEP):
                self.env.render()
                if pobs:
                    print(observation)
                #TODO: if agent is availble, delete the if-else-statement
                if not agent:
                    action = self.env.action_space.sample()
                else:
                    #TODO:use an agent and determine an action from the observation
                    None
                observation, reward, done, info = self.env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
        self.env.close()
