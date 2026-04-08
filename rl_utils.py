import collections
import random
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            np.array(state, dtype=np.float32),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def size(self) -> int:
        return len(self.buffer)


def moving_average(a: List[float], window_size: int) -> np.ndarray:
    if window_size <= 1:
        return np.array(a, dtype=np.float32)
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[: window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_off_policy_agent(
    env,
    agent,
    num_episodes: int,
    replay_buffer: ReplayBuffer,
    minimal_size: int,
    batch_size: int,
    collect_losses: bool = False,
) -> List[float]:
    return_list = []
    loss_history = {
        "critic_1_loss": [],
        "critic_2_loss": [],
        "actor_loss": [],
        "alpha_loss": [],
    }
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f"Iteration {i}") as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0.0
                state, _ = env.reset()
                done = False
                truncated = False
                while not done and not truncated:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done or truncated)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict: Dict[str, np.ndarray] = {
                            "states": b_s,
                            "actions": b_a,
                            "rewards": b_r,
                            "next_states": b_ns,
                            "dones": b_d,
                        }
                        agent.update(transition_dict)
                        if collect_losses and hasattr(agent, "latest_losses"):
                            for key in loss_history:
                                if key in agent.latest_losses:
                                    loss_history[key].append(agent.latest_losses[key])
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": f"{num_episodes / 10 * i + i_episode + 1:.0f}",
                            "return": f"{np.mean(return_list[-10:]):.3f}",
                        }
                    )
                pbar.update(1)
    if collect_losses:
        return return_list, loss_history
    return return_list
