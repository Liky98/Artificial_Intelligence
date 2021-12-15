import time

import gym

env = gym.make("FrozenLake-v1", is_slippery=False)
observation = env.reset()

for _ in range(100):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    time.sleep(1)

    if done:
        observation = env.reset()


env.close()

