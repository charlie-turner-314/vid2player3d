from rl_games.algos_torch import torch_ext

import learning.common_player as common_player

import torch
import os
from tqdm import tqdm

class SelfPlayerPlayer(common_player.CommonPlayer):
    def __init__(self, config):
        super().__init__(config)
        self.task = self.env.task
        self.clip_actions = False

        model_cp = config.get('model_cp', None)
        if model_cp is not None and not config.get('load_checkpoint', False):
            print("Loading v2p policy ...")
            self.load_cps(model_cp)

    def load_cps(self, model_cp:str):
        """
        Load the two models from checkpoint files
        """
        print("LOAD CP")

    def restore(self, cp_name):
        print("RESTORE")
    
    def get_action(self, obs_dict, is_determenistic=False):
        print("GET ACTION")

    def run(self):
        print("RUN")

