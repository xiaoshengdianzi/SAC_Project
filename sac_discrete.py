import random
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import rl_utils


class PolicyNetDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class QValueNetDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SACDiscrete:
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
    ):
        self.actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)

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
        self.policy_update_interval = 2
        self.update_step = 0
        self._last_actor_loss = 0.0
        self._last_alpha_loss = 0.0

    def take_action(self, state):
        state_array = np.asarray(state, dtype=np.float32)
        state = torch.as_tensor(state_array, device=self.device).unsqueeze(0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item())

    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_next = self.target_critic_1(next_states)
        q2_next = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_next, q2_next), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * (1 - dones) * next_value
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        self.update_step += 1
        states = torch.tensor(transition_dict["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32, device=self.device).view(-1, 1)

        td_target = self.calc_target(rewards, next_states, dones)

        q1 = self.critic_1(states).gather(1, actions)
        q2 = self.critic_2(states).gather(1, actions)
        critic_1_loss = F.smooth_l1_loss(q1, td_target.detach())
        critic_2_loss = F.smooth_l1_loss(q2, td_target.detach())

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_2_optimizer.step()

        if self.update_step % self.policy_update_interval == 0:
            probs = self.actor(states)
            log_probs = torch.log(probs + 1e-8)
            entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
            q1_all = self.critic_1(states)
            q2_all = self.critic_2(states)
            min_qvalue = torch.sum(probs * torch.min(q1_all, q2_all), dim=1, keepdim=True)
            actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-10.0, 2.0)

            self._last_actor_loss = float(actor_loss.item())
            self._last_alpha_loss = float(alpha_loss.item())

        self.latest_losses = {
            "critic_1_loss": float(critic_1_loss.item()),
            "critic_2_loss": float(critic_2_loss.item()),
            "actor_loss": self._last_actor_loss,
            "alpha_loss": self._last_alpha_loss,
        }

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


def run_train(seed=0):
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor_lr = 1e-3
    critic_lr = 1e-2
    alpha_lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    target_entropy = -1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "[CartPole config] "
        f"episodes={num_episodes}, actor_lr={actor_lr}, critic_lr={critic_lr}, alpha_lr={alpha_lr}, "
        f"gamma={gamma}, tau={tau}, buffer={buffer_size}, minimal={minimal_size}, batch={batch_size}, "
        f"target_entropy={target_entropy}"
    )

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = SACDiscrete(
        state_dim,
        hidden_dim,
        action_dim,
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
    plt.title(f"Discrete SAC on {env_name}")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"{env_name}_sac_discrete_seed{seed}.png"
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
        values = np.asarray(loss_history[key], dtype=np.float32)
        ax.plot(values, color=color, linewidth=0.9, alpha=0.28, label="raw")
        if len(values) >= 25:
            smooth = rl_utils.moving_average(values.tolist(), 25)
            ax.plot(smooth, color=color, linewidth=1.8, label="ma(25)")
        ax.set_title(title)
        ax.set_xlabel("Update Steps")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    fig.suptitle(f"Discrete SAC Loss Curves on {env_name}")
    plt.tight_layout()
    loss_plot_path = output_dir / f"{env_name}_sac_discrete_losses_seed{seed}.png"
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to: {loss_plot_path}")


if __name__ == "__main__":
    run_train()
