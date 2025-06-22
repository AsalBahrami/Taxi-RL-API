import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from bl_taxi_test import *
import os
import tianshou as ts
from tqdm import tqdm
import seaborn as sns

def main():
    # Create env
    env = gym.make('Taxi-v3', render_mode='ansi')
    state, _ = env.reset()

    # Load the model
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = BLDuelingDQN(state_shape=state_shape,
                       action_shape=action_shape, number_of_nodes=[64, 512])

    model_state_dict = {}
    model_path = os.path.join(os.path.dirname(__file__), 'VEZK_BL')
    for k, v in torch.load(model_path, map_location=torch.device('cpu')).items():
        if not k.startswith('model_old'):
            model_state_dict[k] = v

    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    policy = ts.policy.DQNPolicy(net, optim, 0.1, 0.1, target_update_freq=0)
    policy.load_state_dict(model_state_dict)

    # Value function values
    v_s_values = []

    for run_i in tqdm(range(100)):
        state, _ = env.reset()

        for step in range(100):
            encoded_state = encode(decode(state))
            q_values, _ = policy.model(torch.tensor([encoded_state]))

            v_s = q_values.mean().item()
            v_s_values.append(v_s)

            best_action = torch.argmax(q_values).item()
            state, _, terminated, truncated, _ = env.step(best_action)
            if terminated or truncated:
                break

    # Plot the distribution of V(s)
    sns.kdeplot(v_s_values, fill=True, color="cornflowerblue", alpha=0.5)
    plt.title("Verteilung der Value-Funktion $V(s)$")
    plt.xlabel("Value-Funktion $V(s)$")
    plt.ylabel("Häufigkeit")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
