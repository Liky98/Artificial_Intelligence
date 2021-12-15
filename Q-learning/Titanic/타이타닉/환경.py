""" 환경 구성 """
import pandas as pd

class Environment:
    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.correct_reward = 1
        self.X_reward = -1

    def reset(self):
        self.observation = None

    def loadData(self):
        self.trainData = pd.read_csv('C:/Users/LeeKihoon/PycharmProjects/gachonProject/Q-learning/Titanic/train.csv')
        self.testData = pd.read_csv('C:/Users/LeeKihoon/PycharmProjects/gachonProject/Q-learning/Titanic/test.csv')
        return

    def check_if_reward(self, state) :
        if state == "False" :
            return self.X_reward
        else:
            return self.correct_reward




