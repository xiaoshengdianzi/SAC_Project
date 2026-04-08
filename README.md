<<<<<<< HEAD
# SAC (Soft Actor-Critic) Implementation

This repo contains a minimal SAC implementation for both continuous (Pendulum-v1) and discrete (CartPole-v1) action spaces, with a simple training/evaluation entrypoint.

## Environment

- Python 3.8+
- PyTorch
- gymnasium (preferred) or gym (fallback)
- numpy, tqdm, matplotlib

## Clone

```bash
git clone [your-github-repo-url]
```

## Install

```bash
pip install torch numpy tqdm matplotlib gymnasium
```

If you use gym (older installs), replace gymnasium with gym.

## Run

### Continuous Action Space (Pendulum-v1)
```bash
python main.py --task pendulum
```

### Discrete Action Space (CartPole-v1)
```bash
python main.py --task cartpole
```

Behavior:
- The script trains, plots the return curve and loss curves, then saves the results to the `outputs` directory.

## Results

The following curves are generated during training:

### Pendulum-v1 (Continuous SAC)
- `outputs/Pendulum-v1_sac_seed0.png` - Return curve
- `outputs/Pendulum-v1_sac_losses_seed0.png` - Loss curves

### CartPole-v1 (Discrete SAC)
- `outputs/CartPole-v1_sac_discrete_seed0.png` - Return curve
- `outputs/CartPole-v1_sac_discrete_losses_seed0.png` - Loss curves

## Notes

- Default training is 100 episodes for Pendulum-v1 and 200 episodes for CartPole-v1.
- You can edit hyperparameters inside `run_train()` in `sac_continuous.py` and `sac_discrete.py` if needed.
- You can specify a random seed with `--seed` parameter.

## Project Structure

- `main.py` - Main entrypoint for running experiments
- `sac_continuous.py` - SAC implementation for continuous action spaces
- `sac_discrete.py` - SAC implementation for discrete action spaces
- `rl_utils.py` - Utility functions for reinforcement learning
- `outputs/` - Directory for saving training results and plots
=======
# SAC_Project
Soft Actor-Critic (SAC) implementation in PyTorch. A maximum entropy deep reinforcement learning algorithm for continuous control tasks, featuring automatic alpha tuning and double Q-learning.
>>>>>>> b6ff54239416454d7306f96604cc8c5d1182eaa5
