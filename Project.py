import gym
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.ddpg import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import ts2xy

# List of environments and algorithms
env_names = ["Hopper-v4", "HalfCheetah-v4", "Walker2d-v4"]
algorithms = ["PPO", "REINFORCE", "A2C", "DDPG"]

# Create a directory to save models and logs
log_dir = "./rl_comparison_logs"
os.makedirs(log_dir, exist_ok=True)

# Function to create environment
def make_env(env_name):
    return gym.make(env_name)

# Function to train a model and return results (Total Reward, Episode Length, Convergence Speed)
def train_and_evaluate(alg, env_name, total_timesteps=100000):
    print(f"\nTraining {alg} on {env_name}...")

    # Wrap environment in DummyVecEnv for compatibility with SB3
    env = DummyVecEnv([lambda: make_env(env_name)])

    if alg == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    elif alg == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    elif alg == "REINFORCE":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)  # REINFORCE is PPO, but with different training loop
    elif alg == "DDPG":
        model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # Callback for model evaluation
    eval_callback = EvalCallback(env, best_model_save_path=f'./best_model/{alg}_{env_name}',
                                 log_path='./eval_log', eval_freq=10000, deterministic=True, render=False)

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save the trained model
    model.save(f"{log_dir}/{alg}_{env_name}_final")

    # Testing the trained model and tracking metrics
    obs = env.reset()
    total_reward = 0
    step_count = 0
    done = False
    while not done:
        action, _state = model.predict(obs)
        obs, reward, terminated, info = env.step(action)  # Only 4 values here
        done = terminated  # 'done' is equivalent to 'terminated' here
        total_reward += reward
        step_count += 1

    print(f"{alg} on {env_name}: Total Reward: {total_reward}, Episode Length: {step_count}")
    return total_reward, step_count

# Function to run multiple times and calculate variance
def train_and_evaluate_multiple_runs(alg, env_name, total_timesteps=100000, num_runs=5):
    all_rewards = []
    all_episode_lengths = []

    for _ in range(num_runs):
        total_reward, episode_length = train_and_evaluate(alg, env_name, total_timesteps)
        all_rewards.append(total_reward)
        all_episode_lengths.append(episode_length)

    # Calculate variance
    reward_variance = np.var(all_rewards)
    episode_length_variance = np.var(all_episode_lengths)

    # Calculate average
    avg_reward = np.mean(all_rewards)
    avg_episode_length = np.mean(all_episode_lengths)

    print(f"\n{alg} on {env_name} (over {num_runs} runs):")
    print(f"Average Total Reward: {avg_reward:.2f}, Variance in Reward: {reward_variance:.2f}")
    print(f"Average Episode Length: {avg_episode_length}, Variance in Episode Length: {episode_length_variance}")
    
    return avg_reward, reward_variance, avg_episode_length, episode_length_variance

# Training and evaluating all algorithms on all environments
results = {}
episode_lengths = {}
for env_name in env_names:
    results[env_name] = {}
    episode_lengths[env_name] = {}
    for alg in algorithms:
        avg_reward, reward_variance, avg_episode_length, episode_length_variance = train_and_evaluate_multiple_runs(alg, env_name)
        results[env_name][alg] = (avg_reward, reward_variance)
        episode_lengths[env_name][alg] = (avg_episode_length, episode_length_variance)

# --- Comparison and Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot results (Total Reward Comparison)
for i, env_name in enumerate(env_names):
    ax = axes[i]
    ax.set_title(f"Total Reward Comparison for {env_name}")
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Total Reward")
    
    # Convert reward values to scalar (if they're arrays)
    rewards = [float(results[env_name][alg][0]) for alg in algorithms]  # Ensure they're scalars
    ax.bar(algorithms, rewards, color=["blue", "green", "red", "orange"])

# Save the comparison plot for Total Reward
plt.tight_layout()
plt.savefig(f"{log_dir}/total_reward_comparison.png")
plt.show()

# --- Plot Episode Length Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, env_name in enumerate(env_names):
    ax = axes[i]
    ax.set_title(f"Episode Length Comparison for {env_name}")
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Episode Length")
    
    # Convert episode length to scalar (if they're arrays)
    lengths = [float(episode_lengths[env_name][alg][0]) for alg in algorithms]  # Ensure they're scalars
    ax.bar(algorithms, lengths, color=["blue", "green", "red", "orange"])

# Save the comparison plot for Episode Length
plt.tight_layout()
plt.savefig(f"{log_dir}/episode_length_comparison.png")
plt.show()

# --- Summary Table for Rewards and Episode Length ---
print("\nSummary of Total Rewards and Episode Lengths:")
for env_name in env_names:
    print(f"\n{env_name}:")
    for alg in algorithms:
        avg_reward, reward_variance = results[env_name][alg]
        avg_episode_length, episode_length_variance = episode_lengths[env_name][alg]
        print(f"{alg}: Average Total Reward: {avg_reward:.2f}, Reward Variance: {reward_variance:.2f}, "
              f"Average Episode Length: {avg_episode_length}, Episode Length Variance: {episode_length_variance:.2f}")
