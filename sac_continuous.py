import random
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

import rl_utils


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 1e-6
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        action = torch.tanh(normal_sample)

        # tanh squash 修正项
        log_prob = dist.log_prob(normal_sample)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
    ):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float32, device=device, requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.latest_losses = {}

    def take_action(self, state):
        state_array = np.asarray(state, dtype=np.float32)
        state = torch.as_tensor(state_array, device=self.device).unsqueeze(0)
        action, _ = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def calc_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32, device=self.device).view(-1, 1)

        # Pendulum 奖励重塑，和教材一致
        rewards = (rewards + 8.0) / 8.0

        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = F.mse_loss(self.critic_1(states, actions), td_target.detach())
        critic_2_loss = F.mse_loss(self.critic_2(states, actions), td_target.detach())

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.latest_losses = {
            "critic_1_loss": float(critic_1_loss.item()),
            "critic_2_loss": float(critic_2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
        }

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


def run_train(seed=0):
    env_name = "Pendulum-v1"
    env = gym.make(env_name)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -float(action_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = SACContinuous(
        state_dim,
        hidden_dim,
        action_dim,
        action_bound,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
    )

    return_list, loss_history = rl_utils.train_off_policy_agent(
        env, agent, num_episodes, replay_buffer, minimal_size, batch_size, collect_losses=True
    )

    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 9)

    plt.figure(figsize=(9, 5))
    plt.plot(episodes_list, return_list, color="tab:blue", alpha=0.45, linewidth=1.5, label="Episode Return")
    plt.plot(episodes_list, mv_return, color="tab:orange", linewidth=2.2, label="Moving Avg Return (window=9)")
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"SAC on {env_name}")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"{env_name}_sac_seed{seed}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {plot_path}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()
    loss_keys = [
        ("critic_1_loss", "Critic 1 Loss", "tab:blue"),
        ("critic_2_loss", "Critic 2 Loss", "tab:orange"),
        ("actor_loss", "Actor Loss", "tab:green"),
        ("alpha_loss", "Alpha Loss", "tab:red"),
    ]
    for ax, (key, title, color) in zip(axes, loss_keys):
        ax.plot(loss_history[key], color=color, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Update Steps")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
    fig.suptitle(f"SAC Loss Curves on {env_name}")
    plt.tight_layout()
    loss_plot_path = output_dir / f"{env_name}_sac_losses_seed{seed}.png"
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to: {loss_plot_path}")


if __name__ == "__main__":
    run_train()
