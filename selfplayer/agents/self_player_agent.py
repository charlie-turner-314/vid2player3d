from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common

import learning.common_agent as common_agent 
from utils.common import AverageMeter, get_eta_str

import torch 
import time
import os
import numpy as np
from torch import nn
from tqdm import tqdm


class SelfPlayerAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):
        print("INITIALIZING SELF PLAYER AGENT")
        super().__init__(base_name, config)
        self.task = self.vec_env.env.task
        print("TASK:", self.task.__class__.__name__)
        print("SELF PLAYER AGENT INITIALIZED")
    
    def restore(self, cp_name):
        print("RESTORE")
    
    def load_pretrained(self, cp_name):
        print("LOAD PRETRAINED")
    
    def set_network_weights(self, weights):
        print("SET NETWORK WEIGHTS")
    
    def set_stats_weights(self, weights):
        print("SET STATS WEIGHTS")
    
    def train(self):
        print("TRAIN")
        self.init_tensors()                     # Initialize observation and value tensors
        self.last_mean_rewards = -100500        # Set rewards to a very low value
        self.best_mean_rewards = -100500
        self.obs = self.env_reset()             # Reset the environment and use the observation as the current observation
        self.curr_frames = self.batch_size_envs # NOTE: Not sure what current frames are

        pretrained_model_cp = self.config.get('pretrained_model_cp', None)
        if pretrained_model_cp is not None and not self.config.get('load_checkpoint', False):
            self.load_pretrained(pretrained_model_cp)
        
        # MULTIGPU STUFF WOULD GO HERE

        self._init_train() # Don't think this does anything in this case

        while True: # Train forever? i'd like to do a tqdm here 
            if hasattr(self.task, 'pre_epoch'): 
                self.task.pre_epoch(self.epoch_num) # Do any pre-epoch stuff
            
            epoch_num = self.update_epoch() # Increment the epoch number
            train_info = self.train_epoch() # Train the epoch

            
            print("Epoch Complete")
            break

        print("Training Complete")

    def _eval_critic(self, obs_dict):
        print("EVAL CRITIC")
    
    def play_steps(self) -> dict:
        print("PLAY STEPS")
        self.set_eval() # call .eval() on the pytorch model 

        done_indices = [] # Keep track of terminated environments during each step
        update_list = self.update_list # get the things we need to update

        self.step_rewards = AverageMeter()
        self.step_sub_rewards = AverageMeter() # NOTE: work out what the difference is

        self.obs = self.env_reset(done_indices) # Reset the environments that have terminated
        for n in range(self.horizon_length): # Step horizon_length times
            self.experience_buffer.update_data('obses', n, self.obs['obs']) # store the observations in the experience buffer

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic({'obs': self.obs['obs']})
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.step_rewards.update(rewards.mean(dim=0))
            if infos['sub_rewards'] is not None:
                self.step_sub_rewards.update(infos['sub_rewards'].mean(dim=0))
                self.sub_rewards_names = infos['sub_rewards_names']
            # NOTE: only save the last step's info for simplicity
            self.infos = infos

            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

            self.obs = self.env_reset(done_indices)

            # For debug
            self.task.render_vis()

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict


    
    def calc_gradients(self, input_dict):
        print("CALC GRADIENTS")