import os.path
import json
import pickle
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP


class RLConfig:
    def __init__(
            self,
            n_choose_obj=30,
            n_precision=0,
            run_limit_times=5,
            score_mode=None,
            value_range=None,
            reward_set=0.5,
            learning_rate=0.04,
            read_from_file=False,
            file_path=""
    ):
        if not read_from_file:
            if value_range is None:
                value_range = [-3, 4]
            self.n_choose_obj = n_choose_obj
            self.n_precision = n_precision
            self.run_limit_times = run_limit_times
            self.score_mode = score_mode
            self.value_range = value_range
            self.reward_set = reward_set
            self.learning_rate = learning_rate
        else:
            self.loadRLDetails(file_path)

    def logRLDetails(self, file_path):
        record = {
            "n_choose_obj": self.n_choose_obj,
            "n_precision": self.n_precision,
            "run_limit_times": self.run_limit_times,
            "score_mode": self.score_mode,
            "value_range": self.value_range,
            "reward_set": self.reward_set,
            "learning_rate": self.learning_rate
        }
        with open(file_path, "w") as f:
            json.dump(record, f)

    def loadRLDetails(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                record = json.loads(f.read())
                self.n_choose_obj = record["n_choose_obj"]
                self.n_precision = record["n_precision"]
                self.run_limit_times = record["run_limit_times"]
                self.value_range = record["value_range"]
                self.reward_set = record["reward_set"]
                self.learning_rate = record["learning_rate"]
            return True
        else:
            print(f"{file_path} does not exist")
            return False


class CoordCompute:
    def __init__(
            self,
            map_size,
            edge_reward,
            opponent_score,
            n_choose_obj,
            action_space,
            n_precision,
            value_range
    ):
        self.scene_info = None
        self.map_size = map_size
        self.edge_reward = edge_reward
        self.opponent_score = opponent_score
        self.n_choose_obj = n_choose_obj
        self.action_space = action_space
        self.n_precision = n_precision
        self.value_range = value_range

    def setSceneInfo(self, scene_info):
        self.scene_info = scene_info

    def edgeProcess(self, features):
        self_x = self.scene_info["self_x"]
        self_y = self.scene_info["self_y"]
        self_w = self.scene_info["self_w"]
        self_h = self.scene_info["self_h"]
        all_distance = {
            "LEFT": abs(self_x - self_w / 2 - 25),
            "RIGHT": abs(self.map_size[0] - (self_x + self_w / 2)),
            "UP": abs(self_y - self_h / 2),
            "DOWN": abs(self.map_size[1] - (self_y + self_h / 2))
        }
        for key in all_distance.keys():
            features.append(
                [
                    all_distance[key] + 10,
                    key,
                    self.edge_reward
                ]
            )
        return features

    def preprocessData(self):
        features_list = [0, 0, 0, 0]
        features = []

        opponent_x = self.scene_info["opponent_x"]
        opponent_y = self.scene_info["opponent_y"]
        opponent_lv = self.scene_info["opponent_lv"]

        self_x = self.scene_info["self_x"]
        self_y = self.scene_info["self_y"]
        self_lv = self.scene_info["self_lv"]

        features = self.edgeProcess(features)

        if self_lv > opponent_lv:
            opponent_score = self.opponent_score[0]
        elif self_lv == opponent_lv:
            opponent_score = self.opponent_score[1]
        else:
            opponent_score = self.opponent_score[2]

        opponent_distance = countDistance((self_x, self_y), (opponent_x, opponent_y))
        features.append(
            [
                opponent_distance + 1,
                classifyDirection((self_x, self_y), opponent_x, opponent_y),
                opponent_score
            ]
        )

        for food in self.scene_info["foods"]:
            score = food["score"]
            x = food["x"]
            y = food["y"]
            distance = countDistance((self_x, self_y), (x, y))
            bias_distance = distance + 1
            direction = classifyDirection((self_x, self_y), x, y)
            features.append([bias_distance, direction, score])

        features.sort(key=lambda element: element[0])
        for i in range(min(self.n_choose_obj, len(features))):
            features_list[self.action_space.index(features[i][1])] += features[i][2] / features[i][0]

        for i in range(len(features_list)):
            features_list[i] = parseSpecPrecision(100, features_list[i], self.n_precision)

        # print("=====\n", features_list[:min(self.n_choose_obj, len(features))])
        # features_list = confineRange(features_list, 5, self.n_precision)
        features_list = linearScale(features_list, self.value_range[0], self.value_range[1])
        # print(features_list[:min(self.n_choose_obj, len(features))], "\n=====")
        features.clear()

        return features_list


def parseSpecPrecision(n_amplify, value, n_precision):
    if n_precision == 0:
        return int(value * n_amplify)
    else:
        return round(value * n_amplify, n_precision)


def confineRange(array, max_value, n_precision=0):
    temp_array = [abs(num) for num in array]
    max_num = max(temp_array)
    if max_num > max_value and max_num != 0:
        for i in range(len(array)):
            array[i] = (float(array[i]) / float(max_num)) * max_value
            array[i] = parseSpecPrecision(1, array[i], n_precision)

    return array


def linearScale(value_list, new_min, new_max):
    old_min = min(value_list)
    old_max = max(value_list)
    if old_max == old_min:
        # print("value_list is constant:", value_list)
        if old_max > new_max:
            return [new_max] * len(value_list)
        elif old_max < new_min:
            return [new_min] * len(value_list)
        else:
            return value_list
    if not old_max <= new_max or not old_min >= new_min:
        scaled_values = []
        for value in value_list:
            scaled_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            scaled_value = int(Decimal(scaled_value).quantize(0, ROUND_HALF_UP))
            scaled_values.append(scaled_value)

        return scaled_values
    return value_list


def processPickle(q_table, model_path, mode, chunk_size=10000):
    model_file_prefix = "model_chunk_"
    model_dir = os.path.join(model_path, "q_table_chunks")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if mode == "w":
        try:
            for i in range(0, len(q_table), chunk_size):
                chunk_q_table = q_table.iloc[i:i + chunk_size]
                chunk_file = os.path.join(model_dir, f"{model_file_prefix}{i}.pickle")
                with open(chunk_file, 'wb') as f:
                    pickle.dump(chunk_q_table, f)
            return True
        except IOError as e:
            print(e)
            return False
    elif mode == "r":
        try:
            chunk_files = [f for f in os.listdir(model_dir) if f.startswith(model_file_prefix)]
            if not len(chunk_files) == 0:
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
                q_table = pd.concat(q_tables)
            else:
                q_table = None
            return q_table
        except IOError as e:
            print(e)
            return None


def countDistance(coord1, coord2):
    return np.sqrt(np.square(coord1[0] - coord2[0]) + np.square(coord1[1] - coord2[1]))


def classifyDirection(self_coord, x, y):
    if abs(self_coord[0] - x) < abs(self_coord[1] - y):
        if y < self_coord[1]:
            return "UP"
        else:
            return "DOWN"
    else:
        if x < self_coord[0]:
            return "LEFT"
        else:
            return "RIGHT"

