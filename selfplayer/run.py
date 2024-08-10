from utils.config import set_np_formatting, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from agents.self_player_agent import SelfPlayerAgent
from players.self_player_player import SelfPlayerPlayer

from models.self_player_models import SelfPlayerModel
from models.self_player_builder import SelfPlayerBuilder
from models.v2p_network_builder_dual import V2PBuilderDual


def create_rlgpu_env(**kwargs):
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print("Number of Environments: ", env.num_envs)
    print("Number of Actions:      ", env.num_actions)
    print("Number of observations: ", env.num_obs)
    print("Number of states:       ", env.num_states)

    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env

class RLGPUAlgoObserver(AlgoObserver):
    """
    An AlgoObserver is an object that can be attached to an algorithm to monitor its progress.
    This includes logging, printing, and other forms of output. It does not actually modify the
    algorithm's behavior. 
    """

    def __init__(self, use_successes = True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
    
    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if not (args.tmp or args.no_log):
            if self.consecutive_successes.current_size > 0:
                mean_con_successes = self.consecutive_successes.get_mean()
                self.algo.log_dict.update({'successes/consecutive_successes/mean': mean_con_successes})
        return
    
class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        print("INITIALIZING RLGPU ENV")
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info
    
vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU',
})


def build_alg_runner(algo_observer: RLGPUAlgoObserver) -> Runner:
    runner = Runner(algo_observer)

    # AGENT: The agent is responsible for learning the policy.
    runner.algo_factory.register_builder('selfplayer', lambda **kwargs : SelfPlayerAgent(**kwargs))        # training agent
    runner.player_factory.register_builder('selfplayer', lambda **kwargs : SelfPlayerPlayer(**kwargs))     # testing agent
    runner.model_builder.model_factory.register_builder('selfplayer', lambda network, **kwargs : SelfPlayerModel(network))    # network wrapper
    runner.model_builder.network_factory.register_builder('selfplayer', lambda **kwargs : SelfPlayerBuilder())     # actuall network definition class
    # runner.model_builder.network_factory.register_builder('vid2player_dual', lambda **kwargs : V2PBuilderDual())     # actuall network definition class
    # runner.model_builder.network_factory.register_builder('vid2player_dual_v2', lambda **kwargs : V2PBuilderDualV2())     # actuall network definition class

    # TESTING & VISUALISATION
    # runner.model_builder.network_factory.register_builder('vid2player', lambda **kwargs : V2PBuilder())
    runner.model_builder.network_factory.register_builder('selfplayer', lambda **kwargs : V2PBuilderDual())


    return runner

def main():
    global args
    global cfg
    global cfg_train

    set_np_formatting()
    args = get_args()
    cfg, cfg_train = load_cfg(args)

    vargs = vars(args)

    algo_observer = RLGPUAlgoObserver()

    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    print("* " * 50)
    runner.run(vargs)

    return



if __name__ == "__main__":
    main()