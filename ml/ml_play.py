import json
import os.path
import pickle
import re
import sys
import numpy as np
import pandas as pd
import warnings
sys.path.append(r"D:\PAIA\PAIA-Desktop-win32-x64-2.4.5\resources\app.asar.unpacked\games\swimming-squid-battle\ml")
from RL_brain import QLearningTable
from decimal import Decimal, ROUND_HALF_UP


warnings.simplefilter(action='ignore', category=FutureWarning)


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.source_path = os.path.dirname(__file__)
        self.dataset = ""
        self.model_path = ""
        self.action_space = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
        self.n_actions = len(self.action_space)
        self.loaded_q_table = None
        self._initialize()
        self.RL = QLearningTable(actions=list(range(self.n_actions)),
                                 learning_rate=0.04,
                                 loaded_q_table=self.loaded_q_table)
        self.observation = None
        self.observation_ = None
        self.scene_info = {}
        self.coord = []
        self.n_choose_obj = 20
        self.index = 0
        self.scores = [0, 0]
        self.reward_set = 0.5
        self.n_precision = 0
        self.is_run = False
        self.action = None
        print("Initial ml script")
        self.RL.q_table = self.RL.removeDuplicateStates(self.RL.q_table)
        print(self.RL.q_table)

    def _initialize(self):
        with open(os.path.join(self.source_path, "config.json"), "r") as f:
            config = json.loads(f.read())
            self.dataset = config["dataset"]
        if not os.path.exists(os.path.join(self.source_path, self.dataset)):
            os.mkdir(os.path.join(self.source_path, self.dataset))
        self.model_path = os.path.join(self.source_path, self.dataset)
        # self.processJson("r")
        self.processPickle("r", 5000)
        self.observation = [0, 0, 0, 0]

    def processPickle(self, mode, chunk_size=10000):
        model_file_prefix = "model_chunk_"
        model_dir = os.path.join(self.model_path, "q_table_chunks")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if mode == "w":
            try:
                for i in range(0, len(self.RL.q_table), chunk_size):
                    chunk_q_table = self.RL.q_table.iloc[i:i + chunk_size]
                    chunk_file = os.path.join(model_dir, f"{model_file_prefix}{i}.pickle")
                    with open(chunk_file, 'wb') as f:
                        pickle.dump(chunk_q_table, f)
            except IOError as e:
                print(e)
        elif mode == "r":
            try:
                chunk_files = [f for f in os.listdir(model_dir) if f.startswith(model_file_prefix)]
                q_tables = []
                for chunk_file in chunk_files:
                    chunk_file = os.path.join(model_dir, chunk_file)
                    with open(chunk_file, 'rb') as f:
                        try:
                            q_table_chunk = pickle.load(f)
                            q_tables.append(q_table_chunk)
                        except Exception as e:
                            print("Error occur when loading the file", chunk_file)
                            print(e)
                self.loaded_q_table = pd.concat(q_tables)
                print(self.loaded_q_table.shape, self.loaded_q_table.size)
            except IOError as e:
                print(e)
            print("Load existed q table")

    def processJson(self, mode):
        model_file = os.path.join(self.model_path, "model.json")
        if mode == "w":
            try:
                self.RL.q_table.to_json(model_file)
            except IOError as e:
                print(e)
        elif mode == "r":
            if not os.path.exists(model_file):
                self.loaded_q_table = None
                print("Create new q table")
            else:
                try:
                    print(f"Read from the file {model_file}")
                    self.loaded_q_table = pd.read_json(model_file, encoding="utf-8")
                except IOError as e:
                    print(e)
                print("Load existed q table")

    def processHdf(self, mode):
        model_file = os.path.join(self.model_path, "model.h5")
        if mode == "w":
            try:
                self.RL.q_table.to_hdf(model_file, key="q_table", mode=mode)
            except IOError as e:
                print(e)
        elif mode == "r":
            if not os.path.exists(model_file):
                self.loaded_q_table = None
                print("Create new q table")
            else:
                try:
                    self.loaded_q_table = pd.read_hdf(model_file, key="q_table", mode=mode)
                except IOError as e:
                    print(e)
                print("Load existed q table")

    def _countDistance(self, x, y):
        sx, sy = [self.scene_info["self_x"], self.scene_info["self_y"]]
        return np.sqrt(np.square(sx - x) + np.square(sy - y))

    def confineRange(self, array, max_value):
        temp_array = [abs(num) for num in array]
        max_num = max(temp_array)
        if max_num > max_value and max_num != 0:
            for i in range(len(array)):
                array[i] = (float(array[i]) / float(max_num)) * max_value
                array[i] = self.parseSpecPrecision(1, array[i], self.n_precision)

        return array

    def classifyDirection(self, x, y):
        sx, sy = [self.scene_info["self_x"], self.scene_info["self_y"]]
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

    def parseSpecPrecision(self, n_amplify, value, n_precision):
        if n_precision == 0:
            return int(value * n_amplify)
        else:
            return round(value * n_amplify, n_precision)

    def preprocessData(self):
        features_list = [0, 0, 0, 0]
        features = []
        for food in self.scene_info["foods"]:
            score = food["score"]
            x = food["x"]
            y = food["y"]
            distance = self._countDistance(x, y)
            bias_distance = distance + 1
            direction = self.classifyDirection(x, y)
            features.append([bias_distance, direction, score])

        features.sort(key=lambda x: x[0])
        self.action_space.index(features[0][1])
        for i in range(min(self.n_choose_obj, len(features))):
            features_list[self.action_space.index(features[i][1])] += features[i][2] / features[i][0]

        for i in range(len(features_list)):
            features_list[i] = self.parseSpecPrecision(100, features_list[i], self.n_precision)

        features_list = self.confineRange(features_list, 5)

        features.clear()

        return features_list

    def update(self, scene_info: dict, keyboard: list = [], *args, **kwargs):
        self.scene_info = scene_info
        if not self.is_run:
            self.observation = self.preprocessData()
            self.action = self.RL.choose_action(str(self.observation))
            self.scores[1] = self.scene_info["score"]
            self.is_run = True
        else:
            self.scores[0] = self.scores[1]
            self.scores[1] = self.scene_info["score"]
            self.observation_ = self.preprocessData()
            reward = self.scores[1] - self.scores[0]
            if self.action < len(self.observation):
                additional_reward = self.observation[self.action]
            else:
                additional_reward = 0
            reward += self.reward_set * additional_reward
            # print(reward)
            status = self.scene_info["status"]
            self.RL.learn(str(self.observation), self.action, reward, str(self.observation_))
            self.observation = self.observation_
            # print(self.observation)
            self.action = self.RL.choose_action(str(self.observation))
            if status == "GAME_PASS" or status == "GAME_OVER":
                print(self.RL.q_table)
                print(f"Shape: {self.RL.q_table.shape}")
                self.processPickle("w", 5000)

        return [self.action_space[self.action]]

    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")
        pass