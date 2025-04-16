# Here is the updated code with the requested changes for 10000 timesteps and averaging over 5 runs

import gym
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

# Define environments, algorithms, learning rates, and architectures
env_names = ["Hopper-v4", "HalfCheetah-v4", "Walker2d-v4"]
algorithms = ["PPO", "REINFORCE", "A2C", "DDPG"]
learning_rates = [0.00025, 0.001, 0.0025]
network_architectures = [[64, 64], [256, 256], [400, 300]]

# Directory for saving models and logs
log_dir = "./rl_ablation_study_logs"
os.makedirs(log_dir, exist_ok=True)

# Function to create an environment
def make_env(env_name):
    return gym.make(env_name)

# Training and evaluation function
def train_and_evaluate(alg, env_name, lr, architecture, total_timesteps=10000, num_runs=5):
    env = DummyVecEnv([lambda: make_env(env_name)])
    policy_kwargs = {"net_arch": architecture}
    total_rewards, episode_lengths = [], []

    for _ in range(num_runs):
        if alg == "PPO" or alg == "REINFORCE":  # REINFORCE simulated with PPO settings
            model = PPO("MlpPolicy", env, learning_rate=lr, verbose=0, tensorboard_log=log_dir, policy_kwargs=policy_kwargs)
        elif alg == "A2C":
            model = A2C("MlpPolicy", env, learning_rate=lr, verbose=0, tensorboard_log=log_dir, policy_kwargs=policy_kwargs)
        elif alg == "DDPG":
            model = DDPG("MlpPolicy", env, learning_rate=lr, verbose=0, tensorboard_log=log_dir, policy_kwargs=policy_kwargs)

        # Train the model
        model.learn(total_timesteps) # learns totality of the model 
        obs = env.reset()
        run_rewards, run_lengths = [], []
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            run_rewards.append(rewards[0])
            run_lengths.append(1)  # Each step counts as one

        total_rewards.extend(run_rewards)
        episode_lengths.extend(run_lengths)

    avg_reward = np.mean(total_rewards)
    reward_variance = np.var(total_rewards)
    avg_length = np.mean(episode_lengths)
    length_variance = np.var(episode_lengths)

    return avg_reward, reward_variance, avg_length, length_variance

# Collect and summarize results, prepare for plotting
results = {}
plot_data = {env: {alg: [] for alg in algorithms} for env in env_names}
for env_name in env_names:
    results[env_name] = {}
    print(f"\nSummary of Total Rewards and Episode Lengths:\n{env_name}:")
    for alg in algorithms:
        results[env_name][alg] = {}
        for lr in learning_rates:
            for architecture in network_architectures:
                config_name = f"LR: {lr}, Arch: {architecture}"
                avg_reward, reward_variance, avg_length, length_variance = train_and_evaluate(alg, env_name, lr, architecture)
                results[env_name][alg][config_name] = (avg_reward, reward_variance, avg_length, length_variance)
                plot_data[env_name][alg].append((config_name, avg_reward, avg_length))
                print(f"{alg} - {config_name}: Average Total Reward: {avg_reward:.2f}, Reward Variance: {reward_variance:.2f}, Average Episode Length: {avg_length}, Episode Length Variance: {length_variance:.2f}")

# Plotting results
fig, axs = plt.subplots(len(env_names), 2, figsize=(18, 12))  # Subplots for each environment
for i, env_name in enumerate(env_names):
    for alg in algorithms:
        labels, rewards, lengths = zip(*plot_data[env_name][alg])
        axs[i, 0].bar(labels, rewards, label=f'{alg}')
        axs[i, 1].bar(labels, lengths, label=f'{alg}')

    axs[i, 0].set_title(f'Total Reward Comparison for {env_name}')
    axs[i, 0].set_ylabel('Average Total Reward')
    axs[i, 0].set_xticklabels(labels, rotation=90, ha='right')
    axs[i, 0].legend()

    axs[i, 1].set_title(f'Episode Length Comparison for {env_name}')
    axs[i, 1].set_ylabel('Average Episode Length')
    axs[i, 1].set_xticklabels(labels, rotation=90, ha='right')
    axs[i, 1].legend()

plt.tight_layout()
plt.savefig(f"{log_dir}/detailed_comparison_plots.png")
plt.show()

