import os.path
import sys
import warnings
import datetime

sys.path.append(r"D:\PAIA\PAIA-Desktop-win32-x64-2.4.5\resources\app.asar.unpacked\games\swimming-squid-battle\ml")
from RL_brain import QLearningTable
from data_utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)


class MLPlay(RLConfig):
    def __init__(self, ai_name, *args, **kwargs):
        self.source_path = os.path.dirname(__file__)
        self.dataset = ""
        self.model_path = ""
        self.store_score_model_path = "final_model"
        self.score_mode = None
        self.action_space = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
        self.n_actions = len(self.action_space)
        self.loaded_q_table = None
        self._initialize()

        self.train_config_file = os.path.join(self.source_path, self.dataset, "train_config.json")
        super().__init__(
            read_from_file=True,
            file_path=self.train_config_file
        )
        print(self.score_mode)
        self.RL = QLearningTable(
            actions=list(range(self.n_actions)),
            learning_rate=self.learning_rate,
            loaded_q_table=self.loaded_q_table
        )
        self.observation = None
        self.scene_info = {}
        self.coord = []
        self.index = 0
        self.average_score = 0
        self.run_times = 0
        self.run_limit_times = 5
        self.average_frame_count = 0
        self.full_frame_count = 1400
        self.action = None
        self.opponent_score = (30, 0, -50)
        self.max_opponent_distance = 300
        self.edge_reward = -1
        self.map_size = (650, 600)
        self.threshold_horizontal = 6
        self.threshold_vertical = 5

        self.coord_computer = CoordCompute(
            map_size=self.map_size,
            edge_reward=self.edge_reward,
            opponent_score=self.opponent_score,
            n_choose_obj=self.n_choose_obj,
            action_space=self.action_space,
            n_precision=self.n_precision,
            value_range=self.value_range
        )

        print("Initial ml script")
        self.RL.q_table = self.RL.removeDuplicateStates(self.RL.q_table)
        print(self.RL.q_table)

    def _initialize(self):
        with open(os.path.join(self.source_path, "config.json"), "r") as f:
            config = json.loads(f.read())
            self.dataset = config["dataset"]
            self.score_mode = config["mode"]
        if not os.path.exists(os.path.join(self.source_path, self.dataset)):
            os.mkdir(os.path.join(self.source_path, self.dataset))
        self.model_path = os.path.join(self.source_path, self.dataset)
        self.loaded_q_table = processPickle(None, self.model_path, "r", 5000)
        self.observation = [0, 0, 0, 0]

    def update(self, scene_info: dict, keyboard: list = [], *args, **kwargs):
        self.scene_info = scene_info
        self.coord_computer.setSceneInfo(scene_info)
        self.observation = self.coord_computer.preprocessData()
        self.action = self.RL.choose_action(str(self.observation))
        status = self.scene_info["status"]
        if status == "GAME_PASS" or status == "GAME_OVER":
            if self.score_mode == "score":
                score = self.scene_info["score"]
                frame_count = self.full_frame_count - self.scene_info["frame"]
                self.average_frame_count += frame_count
                self.average_score += score
                self.run_times += 1
                if self.run_times > self.run_limit_times:
                    self.average_score = int(self.average_score / self.run_times)
                    self.average_frame_count = int(self.average_frame_count / self.run_times)
                    self.model_path = os.path.join(self.source_path, self.store_score_model_path)
                    if not os.path.exists(self.model_path):
                        os.mkdir(self.model_path)
                    today = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
                    self.model_path = os.path.join(self.model_path,
                                                   f"model_score_{self.average_score}_frame_{self.average_frame_count}_{today}")
                    if not os.path.exists(self.model_path):
                        os.mkdir(self.model_path)
                    processPickle(self.RL.q_table, self.model_path, "w", 5000)
                    exit(0)

        return [self.action_space[self.action]]

    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")
        pass
