# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from env.tasks.humanoid_smpl_im_mvae import HumanoidSMPLIMMVAE
from utils.tennis_ball_in_estimator import TennisBallInEstimator
from utils.common import get_opponent_env_ids

import torch


class HumanoidSMPLIMMVAEDual(HumanoidSMPLIMMVAE):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._ball_in_estimator = TennisBallInEstimator(
            self.cfg_v2p.get('ball_traj_file', 'vid2player/data/ball_traj/ball_traj_in_dual.npy'))
    
    def create_sim(self):
        super().create_sim()

    def reset(self, reset_actor_reaction_env_ids, reset_ball_env_ids):
        return self._reset_envs(reset_actor_reaction_env_ids, reset_ball_env_ids)

    def _reset_envs(self, reset_actor_reaction_env_ids, reset_ball_env_ids):
        reset_actor_recovery_env_ids = get_opponent_env_ids(reset_actor_reaction_env_ids)
        reset_actor_env_ids = torch.cat([reset_actor_reaction_env_ids, reset_actor_recovery_env_ids])
        if len(reset_actor_env_ids) > 0:
            self._reset_actors(reset_actor_env_ids)
        if len(reset_ball_env_ids) > 0:
            reaction_traj, recovery_traj = self._reset_balls(reset_actor_recovery_env_ids, reset_ball_env_ids)
        
        # HACK: sync ball states update
        contact_env_ids = get_opponent_env_ids(reset_ball_env_ids)
        reset_ball_env_ids = torch.cat([contact_env_ids, reset_ball_env_ids])
        self._reset_env_tensors(reset_actor_env_ids, reset_ball_env_ids)

        if not self.cfg['env']['is_train']:
            # For vis
            self._update_state_from_sim()

        return reaction_traj, recovery_traj

    def _reset_balls(self, reset_actor_recovery_env_ids, reset_ball_env_ids):
        traj_serve = None
        # print("Resetting balls: reset_actor_recovery_env_ids", reset_actor_recovery_env_ids)    # 1
        # print("Resetting balls: reset_ball_env_ids", reset_ball_env_ids)                        # 0
        if len(reset_actor_recovery_env_ids) > 0:
            # set init ball traj for the other player
            # ball root states for 1
            # self._ball_root_states[reset_actor_recovery_env_ids, :3] = torch.FloatTensor([-0.92, -12.2, 1.36]).to(self.device) 
            # print(self._mvae_player._racket_pos[reset_actor_recovery_env_ids])

            self._ball_root_states[reset_actor_recovery_env_ids, :3] = self._mvae_player._racket_pos[reset_actor_recovery_env_ids].clone().detach()
            self._ball_root_states[reset_actor_recovery_env_ids, 0] = 0.92
            self._ball_root_states[reset_actor_recovery_env_ids, 10:13] = torch.FloatTensor([0, 0, 0]).to(self.device)  # Start with -40 spin?
            self._ball_root_states[reset_actor_recovery_env_ids, 3:7] = torch.FloatTensor([0, 0, 0, 1]).to(self.device)  # Start with -40 spin?

            # use random ball velocity
            num_envs = len(reset_actor_recovery_env_ids) # Number of environments to 
            self._ball_root_states[reset_actor_recovery_env_ids, 7] = torch.rand(num_envs).to(self.device) * 4 + -2
            self._ball_root_states[reset_actor_recovery_env_ids, 8] = torch.rand(num_envs).to(self.device) * 4 + 28
            self._ball_root_states[reset_actor_recovery_env_ids, 9] = torch.rand(num_envs).to(self.device) * 3 + 5

        # sample ball traj to start from contact player's racket pos
        # copy the ball state from the contact player and reverse it
        contact_env_ids = get_opponent_env_ids(reset_ball_env_ids) # 1 is the contact player
        traj, ball_states_in, ball_states_out = self._ball_in_estimator.estimate(
            self._ball_root_states[contact_env_ids],
            adjust=self.cfg_v2p.get('adjust_ball'),
            )
        
        # print(f"Resetting balls: ball_states_in (env {reset_ball_env_ids})", ball_states_in)    # 1
        # print(f"Resetting balls: ball_states_out (env {contact_env_ids})", ball_states_out)  # 0

        self._ball_root_states[reset_ball_env_ids] = ball_states_in  
        self._ball_root_states[contact_env_ids] = ball_states_out

        self._has_bounce[reset_ball_env_ids] = 0
        self._bounce_pos[reset_ball_env_ids] = 0
        self._has_racket_ball_contact[reset_ball_env_ids] = 0
        self._ball_pos[reset_ball_env_ids] = self._ball_root_states[reset_ball_env_ids, 0:3]
        self._ball_vel[reset_ball_env_ids] = self._ball_root_states[reset_ball_env_ids, 7:10]

        # same for contact player
        self._has_bounce[contact_env_ids] = 0
        self._bounce_pos[contact_env_ids] = 0
        self._has_racket_ball_contact[contact_env_ids] = 0
        self._ball_pos[contact_env_ids] = self._ball_root_states[contact_env_ids, 0:3]
        self._ball_vel[contact_env_ids] = self._ball_root_states[contact_env_ids, 7:10]

        reaction_traj = traj
        recovery_traj = traj.clone()
        recovery_traj[:, :2] *= -1

        return reaction_traj, recovery_traj

    def render_vis(self, init=True):
        NotImplemented 