import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
# from .bl_taxi_test import BLDuelingDQN, decode, encode
# bl_taxi_test import *
from bl_taxi_test import *
import os
import tianshou as ts
from tqdm import tqdm
import seaborn as sns

def main():
    # Initialize environment
    env = gym.make('Taxi-v3', render_mode='ansi')
    state, _ = env.reset()

    # Load model
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

    # State importance values
    state_importance_values = []

    for run_i in tqdm(range(100)):
        state, _ = env.reset()
        for step in range(100):
            # s = env.state
            encoded_state = encode(decode(state))
            q_values, _ = policy.model(torch.tensor([encoded_state]))
            # best_action = torch.argmax(q_values).item()

            # I(s) function
            i_s = q_values.max().item() - q_values.min().item()
            # i_s = (q_values.max().item() - q_values.min().item()) / q_values.mean().item()
            # v_s = 
            
            state_importance_values.append(i_s)

            # state, _, terminated, truncated, _ = env.step(best_action)

    # data divided into 20 intervals 
    sns.kdeplot(state_importance_values, fill=True, color="mediumseagreen", alpha=0.5)
    # plt.hist(state_importance_values, bins=20, alpha=0.75, edgecolor='black')
    plt.title("Verteilung der Zustandswichtigkeitswerte ($I(s)$)")
    plt.xlabel("Zustandswichtigkeit($I(s)$)")
    plt.ylabel("Häufigkeit")
    plt.grid(True)
    plt.show()

# main()

if __name__ == "__main__":
    main()
