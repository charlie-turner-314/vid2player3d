"""
physics_selfplay_controller.py
Environment for Tennis self-play.
"""
from players.mvae_player import MVAEPlayer
from env.utils.player_builder import PlayerBuilder
from utils.tennis_ball_out_estimator import TennisBallOutEstimator
from utils.common import AverageMeter
from utils.torch_transform import quat_to_rot6d
from utils import torch_utils
from utils.common import get_opponent_env_ids

from typing import Dict, Tuple
import torch
import math
import pdb


class PhysicsSelfPlayController:
    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        self.cfg = cfg
        self.cfg_v2p = self.cfg["env"]["vid2player"]
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self._max_episode_length = self.cfg["env"]["episodeLength"]
        self._enable_early_termination = self.cfg["env"].get(
            "enableEarlyTermination", False
        )
        self._is_train = self.cfg["env"]["is_train"]

        # from BaseTask
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
        self.headless = cfg["headless"]
        self.num_envs = cfg["env"]["numEnvs"]
        self.create_sim()

        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]
        self._epoch_num = 0

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.extras = {}
        self._terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._sub_rewards = None
        self._sub_rewards_names = None

        self._opponent_pos = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._opponent_vel = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._opponent_pos_history = torch.zeros(
            (self.num_envs, self._obs_opponent_history_length, 2),
            device=self.device,
            dtype=torch.float32,
        )

        self._has_init = False
        self._racket_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self._racket_normal = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self._reward_scales = self.cfg_v2p.get("reward_scales", {}).copy()

        # racket body has already been fliped to be the last body 
        self._num_humanoid_bodies = 24
        self._racket_body_id = 24
        self._head_body_id = 13

        self._obs_ball_traj_length = self.cfg_v2p.get("obs_ball_traj_length", 100)
        self._ball_traj = torch.zeros(
            (self.num_envs, 100, 3), device=self.device, dtype=torch.float32
        )
        self._ball_obs = torch.zeros(
            (self.num_envs, self._obs_ball_traj_length, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._bounce_in = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self._ball_out_estimator = TennisBallOutEstimator(
            self.cfg_v2p.get(
                "ball_traj_out_x_file", "vid2player/data/ball_traj_out_x_v0.npy"
            ),
            self.cfg_v2p.get(
                "ball_traj_out_y_file", "vid2player/data/ball_traj_out_y_v0.npy"
            ),
        )

        # estimate bounce for stats
        self._est_bounce_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self._est_bounce_time = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._est_bounce_in = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._est_max_height = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

        # court dims
        if self.cfg_v2p.get("court_min"):
            self._court_min = torch.FloatTensor(self.cfg_v2p["court_min"]).to(
                self.device
            )
            self._court_max = torch.FloatTensor(self.cfg_v2p["court_max"]).to(
                self.device
            )
            self._court_range = self._court_max - self._court_min

        # target
        self._tar_time = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        self._tar_time_total = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        self._tar_action = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )  # 1 swing 0 recovery

        self._reset_reaction_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._reset_recovery_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self._num_reset_reaction = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        self._num_reset = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        self._distance = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

        self._sub_rewards = [None] * self.num_envs
        self._sub_rewards_names = None
        self._mvae_actions = None
        self._res_dof_actions = None
        if not self._is_train or self.cfg_v2p.get("random_walk_in_recovery", False):
            self._mvae_actions_random = torch.zeros(
                (self.num_envs, self._num_mvae_action),
                device=self.device,
                dtype=torch.float32,
            )

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

    def get_action_size(self):
        return self._num_actions

    def get_actor_obs_size(self):
        return self._num_actor_obs

    def create_sim(self):
        # Set the betas
        self.cfg_v2p["smpl_beta"] = [self.cfg_v2p.get("smpl_beta", [0.0] * 10)]

        self.betas = (
            torch.FloatTensor(self.cfg_v2p["smpl_beta"])
            .repeat(self.num_envs, 1)
            .to(self.device)
        )

        # MOTION PLAYERS
        self._mvae_player = MVAEPlayer(
            self.cfg_v2p,
            num_envs=self.num_envs,
            is_train=self._is_train,
            enable_physics=True,
            device=self.device,
        )
        self._smpl = self._mvae_player._smpl


        # EMBODIED POSE PLAYERS
        self._physics_player = PlayerBuilder(self, self.cfg).build_player()
        self._physics_player.task._smpl = self._smpl
        self._physics_player.task._mvae_player = self._mvae_player
        self._physics_player.task._controller = self

        # Size of the Agent's Observations
        # pos x3, vel x3, body_pos 24x3, body_rot 24x6, racket_normal
        # Things that the agent can observe about itself
        self._num_actor_obs = 3 + 3 + 24 * 3 + 24 * 6 + 3

        # NOTE: (latent space size of the mvae)
        self._num_mvae_action = self._num_actions = 32 # Number of outputs for the controller network -> number of inputs for the MVAE

        self._num_res_dof_action = 0
        if self.cfg_v2p.get("add_residual_dof"):
            self._num_res_dof_action += 3
        self._num_actions += self._num_res_dof_action
        if self.cfg_v2p.get("add_residual_root"):
            self._num_actions += 3

        self.cfg["env"]["numActions"] = self.get_action_size() 

        # Num Observations = Actor Observations + Task Observations
        self.cfg["env"]["numObservations"] = (
            self.get_actor_obs_size() 
            + self.get_task_obs_size()
        )


    def get_task_obs_size(self):
        self._num_task_obs = 3 * self.cfg_v2p.get("obs_ball_traj_length", 100)

        # Remove the ball target from the observations
        # if self.cfg_v2p.get("use_random_ball_target", False):
        #     self._num_task_obs += 2  # Remove this line

        # Add opponent's current and past positions (assuming past 10 positions)
        self._obs_opponent_history_length = 10  # You can adjust this value
        self._num_task_obs += self._obs_opponent_history_length * 2  # x and y positions

        # Optionally, add opponent's velocity
        self._include_opponent_velocity = True  # Set to False if not needed
        if self._include_opponent_velocity:
            self._num_task_obs += 2  # x and y velocities

        return self._num_task_obs

    def reset(self, env_ids=None):
        if env_ids is None:
            if self._has_init: 
                return
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)

    def _reset_envs(self, env_ids):
        if len(env_ids) > 0:
            # if you provide env ids, you have to provide both agents on the same court
            assert len(env_ids) % 2 == 0, len(env_ids)
            if self.cfg_v2p.get('serve_from', 'near') == 'near':
                # If we serve from near, set reaction envs to the first and every other env
                reset_actor_reaction_env_ids = env_ids[::2] 
            else:
                # Otherwise, reaction envs is the second and every other env
                reset_actor_reaction_env_ids = env_ids[1::2]
 
            reset_actor_recovery_env_ids = get_opponent_env_ids(reset_actor_reaction_env_ids) # Recovery envs are just the opponents of the reaction envs

            # Set the boolean flags for resetting the reaction and recovery envs
            self._reset_reaction_buf[reset_actor_reaction_env_ids] = 1 # Set the reaction envs to reset
            self._reset_recovery_buf[reset_actor_recovery_env_ids] = 1 # Set the recovery envs to reset
        else:
            # If no env ids are provided, set to empty tensors
            reset_actor_reaction_env_ids = reset_actor_recovery_env_ids = torch.LongTensor([])
        
        reset_reaction_env_ids = self._reset_reaction_buf.nonzero(as_tuple=False).flatten() # Get the reaction envs that need to be reset
        reset_recovery_env_ids = self._reset_recovery_buf.nonzero(as_tuple=False).flatten() # Get the recovery envs that need to be reset
        reset_actor_env_ids = env_ids # Get all the env ids that need to be reset

        if len(env_ids) > 0:
            self._mvae_player.reset_dual(reset_actor_reaction_env_ids, reset_actor_recovery_env_ids)
            self._reset_env_tensors(reset_actor_env_ids)
        
        if len(reset_reaction_env_ids) > 0:
            new_traj = self._physics_player.task.reset(
                reset_actor_reaction_env_ids, reset_reaction_env_ids)
            self._ball_traj[reset_reaction_env_ids, :new_traj.shape[1]] = new_traj.to(self.device)
            # mirror ball traj x and y for the opponents
            opponents = get_opponent_env_ids(reset_reaction_env_ids)
            self._ball_traj[opponents] = self._ball_traj[reset_reaction_env_ids] * torch.FloatTensor([-1, -1, 1]).to(self.device)

            
            if not self.headless and not self._has_init:
                self._physics_player.task.render_vis(init=True)

            self._update_state()
        
        if len(reset_recovery_env_ids) > 0:
            self._reset_recovery_tasks(reset_recovery_env_ids)
        if len(reset_reaction_env_ids) > 0:
            self._reset_reaction_tasks(reset_reaction_env_ids, reset_actor_reaction_env_ids)
            self._compute_observations(reset_reaction_env_ids)
        
        self._has_init = True

    def _reset_env_tensors(self, env_ids):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._reset_reaction_buf[env_ids] = 0
        self._reset_recovery_buf[env_ids] = 0
        self._num_reset_reaction[env_ids] = 0
        self._distance[env_ids] = 0

    def _reset_reaction_tasks(self, env_ids, humanoid_env_ids=None):
        if self.cfg_v2p.get('use_history_ball_obs'):
            self._ball_obs[env_ids]     = self._ball_pos[env_ids].view(-1, 1, 3).repeat(1, self._obs_ball_traj_length, 1)
        self._tar_time[env_ids]         = 0
        self._tar_action[env_ids]   = 1
        self._num_reset_reaction[env_ids] += 1
        self._bounce_in[env_ids] = 0

    def _reset_recovery_tasks(self, env_ids):
        self._tar_action[env_ids] = 0
        self._physics_player.task._has_bounce[env_ids] = 0
        self._physics_player.task._bounce_pos[env_ids] = 0

    def pre_physics_step(self, actions):
        self._actions = actions.clone()
        self._mvae_actions = actions[
            :, : self._num_mvae_action
        ].clone() * self.cfg_v2p.get("vae_action_scale", 1.0)

        if self.cfg_v2p.get("random_walk_in_recovery", False):
            in_recovery = self._tar_action == 0
            num_in_recovery = in_recovery.sum()
            self._mvae_actions_random[:num_in_recovery].normal_(0, 1)
            self._mvae_actions_random[:num_in_recovery] = torch.clamp(
                self._mvae_actions_random[:num_in_recovery], -5, 5
            )
            self._mvae_actions[in_recovery] = self._mvae_actions_random[
                :num_in_recovery
            ]

        self._res_dof_actions = torch.empty(0)
        if self.cfg_v2p.get("add_residual_dof"):
            self._res_dof_actions = actions[
                :,
                self._num_mvae_action : self._num_mvae_action
                + self._num_res_dof_action,
            ].clone() * self.cfg_v2p.get("residual_dof_scale", 0.1)
        self._mvae_player.step(self._mvae_actions, self._res_dof_actions)
        if self.cfg_v2p.get("add_residual_root"):
            self._res_root_actions = actions[
                :,
                self._num_mvae_action
                + self._num_res_dof_action : self._num_mvae_action
                + self._num_res_dof_action
                + 3,
            ].clone() * self.cfg_v2p.get("residual_root_scale", 0.02)

        self._physics_player.task.post_mvae_step()

    def _update_state(self):
        self._root_pos = self._physics_player.task._root_pos
        self._root_vel = self._physics_player.task._root_vel
        if not self._is_train:
            self._joint_rot = self._physics_player.task._joint_rot

        self._racket_pos = self._physics_player.task._racket_pos
        self._racket_vel = self._physics_player.task._racket_vel
        self._racket_normal = self._physics_player.task._racket_normal

        self._ball_pos = self._physics_player.task._ball_pos
        self._ball_vel = self._physics_player.task._ball_vel
        self._ball_vspin = self._physics_player.task._ball_vspin

        court_min = [-4.11, 0]
        court_max = [4.11, 11.89]
        serve_min = [0.0, 0]
        serve_max = [4.11, 6.4]
        update_true_bounce = (
            self._tar_action == 0
        ) & self._physics_player.task._has_bounce_now
        bounce_pos = self._physics_player.task._bounce_pos[update_true_bounce]
        self._bounce_in[update_true_bounce] = (
            (bounce_pos[:, 0] > court_min[0])
            & (bounce_pos[:, 0] < court_max[0])
            & (bounce_pos[:, 1] > court_min[1])
            & (bounce_pos[:, 1] < court_max[1])
        )

        # estimate bounce position
        has_contact_now = self._physics_player.task._has_racket_ball_contact_now
        if self.cfg_v2p.get("dual_mode") and has_contact_now.sum() > 0:
            env_ids_contact = has_contact_now.nonzero(as_tuple=False).flatten()
            if env_ids_contact.numel() > 0:
                # Get the ball root states for the environments where contact occurred
                ball_states = self._physics_player.task._ball_root_states[env_ids_contact]
                # Estimate the bounce positions
                has_valid_contact, bounce_pos, bounce_time, max_height = (
                    self._ball_out_estimator.estimate(ball_states)
                )
                if has_valid_contact.sum() > 0:
                    # Get the valid env_ids
                    valid_env_ids = env_ids_contact[has_valid_contact]
                    # Get the opponent env_ids
                    opponent_env_ids = get_opponent_env_ids(valid_env_ids)
                    # Update the estimated bounce positions for the opponent
                    self._est_bounce_pos[opponent_env_ids, :2] = bounce_pos
                    self._est_bounce_time[opponent_env_ids] = bounce_time
                    self._est_max_height[opponent_env_ids] = max_height

                    self._est_bounce_in[opponent_env_ids] = (
                        (self._est_bounce_pos[opponent_env_ids, 0] > court_min[0])
                        & (self._est_bounce_pos[opponent_env_ids, 0] < court_max[0])
                        & (self._est_bounce_pos[opponent_env_ids, 1] > court_min[1])
                        & (self._est_bounce_pos[opponent_env_ids, 1] < court_max[1])
                    )

        self._phase_pred = self._mvae_player._phase_pred

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)


        actor_obs = self._compute_actor_obs(env_ids)
        if torch.isnan(actor_obs).any():
            print("Found NAN in actor obersavations")
            pdb.set_trace()

        task_obs = self._compute_task_obs(env_ids)
        if torch.isnan(task_obs).any():
            print("Found NAN in task obersavations")
            pdb.set_trace()

        obs = torch.cat([actor_obs, task_obs], dim=-1)
        self.obs_buf[env_ids] = obs

    def _compute_actor_obs(self, env_ids):
        player = self._physics_player.task
        actor_obs = torch.cat(
            [
                player._root_pos[env_ids],
                player._root_vel[env_ids],
                (
                    player._rigid_body_pos[env_ids, 1:]
                    - player._root_pos[env_ids].unsqueeze(-2)
                ).view(-1, 24 * 3),
                quat_to_rot6d(
                    player._rigid_body_rot[env_ids, : self._num_humanoid_bodies].view(
                        -1, 4
                    )
                ).view(-1, 24 * 6),
                player._racket_normal[env_ids],
            ],
            dim=-1,
        )
        return actor_obs

    def _compute_task_obs(self, env_ids):
        # Existing ball observations
        self._ball_obs[env_ids] = self._ball_obs[env_ids].roll(-1, dims=1)
        self._ball_obs[env_ids, -1] = self._ball_pos[env_ids].clone()

        if self.cfg_v2p.get("use_history_ball_obs", False):
            ball_obs = self._ball_obs[env_ids]
        else:
            ball_obs = self._ball_traj[env_ids, : self._obs_ball_traj_length]

        # Relative ball position to current racket
        task_obs = ball_obs - self._physics_player.task._rigid_body_pos[env_ids, self._racket_body_id].unsqueeze(-2)

        # Remove ball target from observations
        # if self.cfg_v2p.get("use_random_ball_target", False):
        #     target = self._target_bounce_pos[env_ids, :2] - self._physics_player.task._root_pos[env_ids, :2]
        #     task_obs = torch.cat([task_obs.view(len(env_ids), -1), target], dim=-1)

        # Include opponent's observations
        self._compute_opponent_obs(env_ids)

        # Flatten ball observations
        task_obs = task_obs.view(len(env_ids), -1)

        # Add opponent's position history
        opponent_pos_history = self._opponent_pos_history[env_ids].view(len(env_ids), -1)

        # Optionally, include opponent's velocity
        if self._include_opponent_velocity:
            opponent_vel = self._opponent_vel[env_ids]
            task_obs = torch.cat([task_obs, opponent_pos_history, opponent_vel], dim=-1)
        else:
            task_obs = torch.cat([task_obs, opponent_pos_history], dim=-1)

        return task_obs
    

    def _compute_opponent_obs(self, env_ids):
        """
        Update observations of the opponent agent.
        - Opponent position history (x and y)
        - Opponent velocity (x and y)
        """
        # Assuming opponent's environment IDs are interleaved with the agent's IDs
        opponent_env_ids = get_opponent_env_ids(env_ids)

        # Get opponent's current position (x and y)
        opponent_root_pos = self._physics_player.task._root_pos[opponent_env_ids, :2]
        opponent_root_vel = self._physics_player.task._root_vel[opponent_env_ids, :2]

        # Update opponent position history
        self._opponent_pos_history[env_ids] = self._opponent_pos_history[env_ids].roll(-1, dims=1)
        self._opponent_pos_history[env_ids, -1] = opponent_root_pos

        # Update current opponent position and velocity
        self._opponent_pos[env_ids] = opponent_root_pos
        self._opponent_vel[env_ids] = opponent_root_vel

    def physics_step(self):
        self._physics_player.run_one_step()

        self._ball_traj = self._ball_traj.roll(-1, dims=1)
        self._ball_traj[:, -1] = 0



    def _compute_reward(self, actions):
        reward_weights = self.cfg_v2p.get("reward_weights", {})

        # Compute base rewards (e.g., contact, distance)
        pos_reward = compute_pos_reward(
            self._racket_pos,
            self._phase_pred,
            self._mvae_player._swing_type_cycle,
            self._ball_pos,
            self._physics_player.task._has_racket_ball_contact,
            self._reward_scales,
            reward_weights,
        )
        # Get 'miss' and 'out' from _compute_reset()
        miss = self._reset_recovery_buf & ~self._physics_player.task._has_racket_ball_contact
        out = (self._tar_action == 0) & self._physics_player.task._has_bounce & ~self._bounce_in

        # Compute win and lose rewards
        win_reward, lose_penalty = compute_win_reward_and_lose_penalty(
            self.reset_buf,
            miss,
            out,
            reward_weights,
        )

        # Reward for ball bouncing in court (estimated)
        bounce_in_reward = compute_bounce_in_reward(
            self._est_bounce_in,
            reward_weights,
        )

        # Reward for ball bouncing near the edges of the court (estimated)
        bounce_pos_reward = compute_bounce_pos_reward(
            self._est_bounce_pos,
            self._court_min,
            self._court_max,
            reward_weights,
        )

        # Total reward
        self.rew_buf[:] = pos_reward + win_reward + lose_penalty + bounce_in_reward + bounce_pos_reward
 
        # Update sub_rewards and sub_rewards_names
        self._sub_rewards = torch.stack(
            [pos_reward, win_reward, lose_penalty, bounce_in_reward, bounce_pos_reward], dim=-1
        )

        self._sub_rewards_names = "pos_reward,win_reward,lose_penalty,bounce_in_reward,bounce_pos_reward"

    def _compute_reset(self):
        has_contact = self._physics_player.task._has_racket_ball_contact
        has_bounce = self._physics_player.task._has_bounce
        in_recovery = self._tar_action == 0

        miss_ball = self._ball_pos[:, 1] < self._root_pos[:, 1] - 1
        ball_bounce_twice = has_bounce & (self._ball_pos[:, 2] < 0.05)
        self._reset_recovery_buf = (self._tar_action == 1) & \
            (has_contact | miss_ball | ball_bounce_twice)
        
        self._compute_stats()

        # recovery also marks the reaction of their opponent
        self._reset_reaction_buf[::2] = self._reset_recovery_buf[1::2]
        self._reset_reaction_buf[1::2] = self._reset_recovery_buf[::2]

        miss = self._reset_recovery_buf & ~has_contact
        out = in_recovery & has_bounce & ~self._bounce_in
        terminate = miss | out
        if terminate.sum() > 0:
            terminate[::2] |= terminate[1::2]
            terminate[1::2] |= terminate[::2]

            self.reset_buf[terminate] = 1
            self._reset_reaction_buf[terminate] = 0
            self._reset_recovery_buf[terminate] = 0

    def _compute_stats(self):
        self._distance += (self._root_vel[:, :2]).norm(dim=-1)

    def post_physics_step(self):
        self._tar_time += 1
        self.progress_buf += 1

        self._update_state()
        self._compute_reward(self._actions)
        self._compute_observations()
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf
        self.extras["sub_rewards"] = self._sub_rewards
        self.extras["sub_rewards_names"] = self._sub_rewards_names

    def step(self, actions):
        self.pre_physics_step(actions)

        self.physics_step()

        self.post_physics_step()

    def get_aux_losses(self, model_res_dict):
        aux_loss_specs = self.cfg_v2p.get("aux_loss_specs", dict())
        aux_losses = {}
        aux_losses_weighted = {}

        # default angle to be close to 0
        dof_res = model_res_dict["mus"][
            :, self._num_mvae_action : self._num_mvae_action + self._num_res_dof_action
        ]
        dof_res_loss = (dof_res**2).sum(dim=-1).mean()
        aux_losses["aux_dof_res_loss"] = dof_res_loss
        aux_losses_weighted["aux_dof_res_loss"] = (
            aux_loss_specs.get("dof_res", 0) * dof_res_loss
        )

        return aux_losses, aux_losses_weighted

    def render_vis(self):
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def check_out_of_court(root_pos, court_min, court_max):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    out_of_court = (
        (root_pos[:, 0] < court_min[0])
        .logical_or(root_pos[:, 1] < court_min[1])
        .logical_or(root_pos[:, 0] > court_max[0])
        .logical_or(root_pos[:, 1] > court_max[1])
    )

    return out_of_court.long()


@torch.jit.script
def compute_pos_reward(
    racket_pos: torch.Tensor,
    phase: torch.Tensor,
    swing_type: torch.Tensor,
    ball_pos: torch.Tensor,
    has_contact: torch.Tensor,
    scales: Dict[str, float],
    weights: Dict[str, float],
) -> torch.Tensor:
    # print("racket_pos", racket_pos.tolist())
    # print("phase", phase.tolist())
    # print("swing_type", swing_type.tolist())
    # print("ball_pos", ball_pos.tolist())
    # print("has_contact", has_contact.tolist())
    # print("scales", scales)
    # print("weights", weights)

    w_pos = weights.get("pos", 0.0)

    # position
    pos_diff = ball_pos - racket_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

    # bh contact tends to be earlier
    contact_phase = torch.where(
        (swing_type >= 2),
        torch.ones_like(phase) * 3,
        torch.ones_like(phase) * math.pi,
    )
    phase_diff_rea = phase - contact_phase
    phase_err_rea = phase_diff_rea * phase_diff_rea

    pos_reward = ~has_contact * torch.exp(
        -scales.get("pos", 5.0) * pos_err
    ) * torch.exp(
        -scales.get("phase", 10.0) * phase_err_rea
    ) + has_contact * torch.ones_like(
        pos_err
    )

    # all rewards
    pos_reward = w_pos * pos_reward
    # print("pos_rewards:", pos_reward.tolist())

    return pos_reward

@torch.jit.script
def compute_win_reward_and_lose_penalty(
    reset_buf: torch.Tensor,
    miss: torch.Tensor,
    out: torch.Tensor,
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the win and lose rewards for the agent."""
    # Agent won the point if the opponent hit out
    won_point = reset_buf & out

    # Agent lost the point if they missed the ball
    lost_point = reset_buf & miss

    # Scaling factors
    win_scale = weights.get("win", 10.0)
    lose_scale = weights.get("lose", 10.0)

    # Compute rewards
    win_reward = won_point.float() * win_scale
    lose_penalty = lost_point.float() * (-lose_scale)

    return win_reward, lose_penalty

@torch.jit.script
def compute_bounce_in_reward(
    est_bounce_in: torch.Tensor,
    weights: Dict[str, float],
) -> torch.Tensor:
    """Compute the reward for the ball bouncing within the court.

    Args:
        est_bounce_in (torch.Tensor): Tensor indicating if the estimated bounce is within the court (True/False).
        weights (Dict[str, float]): Dictionary containing scaling factors for rewards.

    Returns:
        torch.Tensor: Tensor of bounce-in rewards for each environment.
    """
    # Scaling factor for bounce-in reward
    bounce_in_scale = weights.get("bounce_in", 5.0)

    # Reward is bounce_in_scale if the ball bounces in, else zero
    bounce_in_reward = est_bounce_in.float() * bounce_in_scale
    

    return bounce_in_reward

@torch.jit.script
def compute_bounce_pos_reward(
    est_bounce_pos: torch.Tensor,
    court_min: torch.Tensor,
    court_max: torch.Tensor,
    weights: Dict[str, float],
) -> torch.Tensor:
    """Compute the reward based on the estimated bounce position within the court.

    Args:
        est_bounce_pos (torch.Tensor): Estimated bounce positions (x, y, z) for each environment.
        court_min (torch.Tensor): Minimum x and y coordinates of the court.
        court_max (torch.Tensor): Maximum x and y coordinates of the court.
        weights (Dict[str, float]): Dictionary containing scaling factors for rewards.

    Returns:
        torch.Tensor: Tensor of bounce position rewards for each environment.
    """
    # Scaling factor for bounce position reward
    bounce_pos_scale = weights.get("bounce_pos", 1.0)

    # Normalize bounce positions to range [0, 1] within the court
    court_range = court_max - court_min  # (x_range, y_range)
    est_bounce_pos_normalized = (est_bounce_pos[:, :2] - court_min) / court_range

    # Distance to the center (0.5, 0.5) represents central court; farther means closer to edges
    center = torch.tensor([0.5, 0.5], device=est_bounce_pos.device)
    distance_to_center = torch.norm(est_bounce_pos_normalized - center, dim=-1)

    # reward is 0 if outside the court. 1 if at the edge. 0 if at the center
    bounce_pos_reward = (1 - distance_to_center) * bounce_pos_scale

    # Clip the reward to be non-negative
    bounce_pos_reward = torch.clamp(bounce_pos_reward, min=0)

    return bounce_pos_reward