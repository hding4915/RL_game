import json
import os.path
import random
from pprint import pprint
import orjson
import pygame
import sys
import numpy as np
import pandas as pd
from RL_brain import QLearningTable


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.source_path = os.path.dirname(__file__)
        self.data_set = ""
        self.model_path = ""
        self.action_space = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
        self.n_actions = len(self.action_space)
        self.loaded_q_table = None
        self._initialize()
        self.RL = QLearningTable(actions=list(range(self.n_actions)), loaded_q_table=self.loaded_q_table)
        self.observation = None
        self.observation_ = None
        self.scene_info = {}
        self.coord = []
        self.n_choose_obj = 8
        self.index = 0
        self.scores = [0, 0]
        self.is_run = False
        self.action = None
        print("Initial ml script")

    def _initialize(self):
        with open(os.path.join(self.source_path, "config.json"), "r") as f:
            config = json.loads(f.read())
            self.data_set = config["dataset"]
        if not os.path.exists(os.path.join(self.source_path, self.data_set)):
            os.mkdir(os.path.join(self.source_path, self.data_set))
        self.model_path = os.path.join(self.source_path, self.data_set, "model.pickle")
        if not os.path.exists(self.model_path):
            self.loaded_q_table = None
        else:
            self.loaded_q_table = pd.read_pickle(self.model_path)

    def _countDistance(self, s, x, y):
        sx = s[0]
        sy = s[1]
        return np.sqrt(np.square(sx - x) + np.square(sy - y))

    def classifyDirection(self, s, x, y):
        sx = s[0]
        sy = s[1]
        if abs(sx - x) < abs(sy - y):
            if y < sy:
                return "UP"
            else:
                return "DOWN"
        else:
            if x < sx:
                return "LEFT"
            else:
                return "RIGHT"

    def preprocessData(self):
        features_dict = {
            "UP": 0,
            "DOWN": 0,
            "LEFT": 0,
            "RIGHT": 0
        }

        for food in self.scene_info["foods"]:
            score = food["score"]
            x = food["x"]
            y = food["y"]
            distance = self._countDistance(self.observation, x, y)
            direction = self.classifyDirection(self.observation, x, y)
            features_dict[direction] += score / distance

        return features_dict

    def update(self, scene_info: dict, keyboard: list = [], *args, **kwargs):
        self.scene_info = scene_info
        if not self.is_run:
            self.observation = self.preprocessData()
            self.action = self.RL.choose_action(self.observation)
            self.scores[1] = self.scene_info["score"]
            self.is_run = True
        else:
            self.scores[0] = self.scores[1]
            self.scores[1] = self.scene_info["score"]
            self.observation_ = self.preprocessData()
            reward = self.scores[1] - self.scores[0]
            status = self.scene_info["status"]
            self.RL.learn(self.observation, self.action, reward, self.observation_)
            self.observation = self.observation_
            self.action = self.RL.choose_action(self.observation)
            if status == "GAME_PASS" or status == "GAME_OVER":
                self.storeModel()

        return self.action

    def storeModel(self):
        self.RL.q_table.to_pickle(self.model_path)

    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")
        pass
