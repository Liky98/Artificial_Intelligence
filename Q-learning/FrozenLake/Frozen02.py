import random
import time
import gym
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery =False) #미끄러짐 false

discount_facter = 0.95 #미래보상 할인계수
epsilon = 0.5 #새로운 액션을 취할 확률
epsilon_decay_factor = 0.999
learning_rate = 0.8 # 기존의 액션과 새로운 액션의 학습 비율
num_episodes = 10000

q_table = np.zeros([env.observation_space.n, env.action_space.n]) #테이블 만들어서 업데이트. [환경상태, 액션]
print(q_table.shape) # 4*4의 크기

for i in range(num_episodes) :
    state = env.reset()
    epsilon = epsilon * epsilon_decay_factor #앱실론은 점점 줄어들음
    done = False

    while not done :
        if np.random.random() < epsilon :
            action = env.action_space.sample() #새로운 액션 찾아서 ㄱㄱ
        else :
            action = np.argmax(q_table[state, : ]) #기존 액션 ㄱㄱ

        new_state, reward, done, _ = env.step(action) #새로운 보상 업데이트
        #Q함수 업데이트
        q_table[state, action] += learning_rate*(reward + discount_facter * np.max(q_table[new_state, :]) - q_table[state,action])
        state = new_state
        if i>9980 :
            env.render()
            print("new_state : ", new_state, " reward : ", reward, " done : ", done)
