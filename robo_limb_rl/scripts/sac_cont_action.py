# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from robo_limb_rl.envs.limb_env import LimbEnv
from robo_limb_rl.arch.Actor_Critic_net import RLAgent
from robo_limb_rl.utils.utils import TrajReplayBuffer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 2  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    
    #Config file path
    config_path: str = "./yaml/default_limb_env.yml"
    """the path to the config file"""
    head_type: str = 'seq2seq_encoder'
    """the type of the head network"""
    freeze_head: bool = False
    """if toggled, the head network will be frozen"""
    pretrained_model: bool = False
    """the path to the pretrained model"""
    clip_norm: float = 1.0
    """max norm for gradient clipping"""


def make_env(env_id, seed, idx, capture_video, run_name, config_path):
    def thunk():
        env = gym.make(env_id, seed=seed, config_path=config_path, render_mode=None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name, args.config_path)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    template_env = gym.make(args.env_id, config_path=args.config_path, render_mode=None)
    hidden_dim = template_env.unwrapped.hidden_dim
    num_layers = template_env.unwrapped.num_layers
    included_power = template_env.unwrapped.include_power_calc
    if args.pretrained_model:
        pretrained_model_path = template_env.model_path
    else:
        pretrained_model_path = None
    
    # ALGO LOGIC: initialize agent here:
    target_agent = RLAgent(envs.single_observation_space,
                           envs.single_action_space,
                           head_type=args.head_type,
                           agent='SAC',
                           hidden_dim=hidden_dim,
                           num_layers=num_layers,
                           batch_size=args.batch_size,
                           freeze_head=args.freeze_head,
                           pretrained_model=pretrained_model_path,
                           device=device).to(device)
    agent = RLAgent(envs.single_observation_space,
                    envs.single_action_space,
                    head_type=args.head_type,
                    agent='SAC',
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    freeze_head=args.freeze_head,
                    batch_size=args.batch_size,
                    pretrained_model=pretrained_model_path,
                    device=device).to(device)
    
    target_agent.load_state_dict(agent.state_dict())
    if args.freeze_head:
        for param in agent.head.parameters():
            param.requires_grad = False
        q_optimizer = optim.Adam(list(agent.critic1.parameters()) 
                               + list(agent.critic2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(agent.actor.parameters()), lr=args.policy_lr)
    else:
        q_optimizer = optim.Adam(list(agent.critic1.parameters()) 
                               + list(agent.critic2.parameters())
                               + list(agent.head.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(agent.actor.parameters())
                                   + list(agent.head.parameters()), lr=args.policy_lr)
    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    
    if args.head_type == 'mlp' or args.head_type == None:
            rb = ReplayBuffer(
                args.buffer_size,
                envs.single_observation_space,
                envs.single_action_space,
                device,
                handle_timeout_termination=False,
            )
    else:
        rb = TrajReplayBuffer(max_size=args.buffer_size,
                              original_obs_space_size=6,
                              new_obs_space_size=np.prod(envs.single_observation_space.shape) - 6,
                              action_space_size=np.prod(envs.single_action_space.shape),
                              max_seq_len=500,
                              device=device)
    

    start_time = time.time()
    os.makedirs(f"../policies/{run_name}", exist_ok=True)
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in tqdm(range(args.total_timesteps)):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = agent.get_action(torch.Tensor(obs).unsqueeze(1).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
                
        # if out of memory, pop the first element
        rb.add(obs, real_next_obs, actions, rewards, terminations, truncations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = agent.get_action(data.next_observations)
                qf1_next_target, qf2_next_target = target_agent.forward_critic(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
            
            qf1_a_values, qf2_a_values = agent.forward_critic(data.observations, data.actions)
            qf1_a_values, qf2_a_values = qf1_a_values.view(-1), qf2_a_values.view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_optimizer.param_groups[0]['params'], args.clip_norm)
            q_optimizer.step()
            
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = agent.get_action(data.observations)
                    qf1_pi, qf2_pi = agent.forward_critic(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_optimizer.param_groups[0]['params'], args.clip_norm)
                    actor_optimizer.step()
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = agent.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        torch.nn.utils.clip_grad_norm_(a_optimizer.param_groups[0]['params'], args.clip_norm)
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(agent.critic1.parameters(), target_agent.critic1.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(agent.critic2.parameters(), target_agent.critic2.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(agent.head.parameters(), target_agent.head.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                    
        if (global_step + 1) % 10000 == 0:
            model_path = f"../policies/{run_name}/agent_{args.exp_name}.cleanrl_model"
            torch.save((agent.state_dict()), model_path)
            print(f"model saved to {model_path}")
    model_path = f"../policies/{run_name}/agent_{args.exp_name}.cleanrl_model"
    torch.save((agent.state_dict()), model_path)
    print(f"model saved to {model_path}")
    
    envs.close()
    writer.close()