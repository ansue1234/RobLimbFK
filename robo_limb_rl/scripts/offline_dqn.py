# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
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
from torch.utils.tensorboard import SummaryWriter
from robo_limb_rl.arch.Q_net import QNet_MLP, QNet_LSTM
from robo_limb_rl.utils.utils import DataLoader


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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    train_frequency: int = 10
    """the frequency of training"""
    network_type: str = "mlp"
    """the type of Q network"""
    state_path: str = None
    """the path of the state data"""
    next_state_path: str = None
    """the path of the next state data"""
    action_path: str = None
    """the path of the action data"""
    reward_path: str = None
    """the path of the reward data"""
    shuffle: bool = True
    """whether to shuffle the data"""


# ALGO LOGIC: initialize agent here:
if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
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
    data_loader = DataLoader(args.state_path, args.next_state_path, args.action_path, args.reward_path, 21, args.batch_size, device)
    obs_space, action_space = data_loader.get_dims()
    # env setup
    if args.network_type == "mlp":
        q_network = QNet_MLP(input_dim=obs_space, output_dim=action_space).to(device)
        target_network = QNet_MLP(input_dim=obs_space, output_dim=action_space).to(device)
    else:
        q_network = QNet_LSTM(input_dim=obs_space, output_dim=action_space).to(device)
        target_network = QNet_LSTM(input_dim=obs_space, output_dim=action_space).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network.load_state_dict(q_network.state_dict())

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook

        # ALGO LOGIC: training.
        if global_step % args.train_frequency == 0:
            obs, next_obs, action, rewards, dones = data_loader.get_data(args.shuffle)
            with torch.no_grad():
                target_max, _ = target_network(next_obs).max(dim=1)
                td_target = rewards + args.gamma * target_max
            old_val = q_network(obs).gather(1, data_loader.get_action(action)).squeeze()
            loss = F.mse_loss(td_target, old_val)
            
            # Supervised loss
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update target network
        if global_step % args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(
                    args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        
    writer.close()