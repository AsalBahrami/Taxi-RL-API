import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from bl_taxi_test import *
import os
import tianshou as ts
from tqdm import tqdm
import seaborn as sns


# NOTE: uncomment for normalization
# def z_score_normalize(array):
#     mean = np.mean(array)
#     std = np.std(array)
#     return (array - mean) / (std + 1e-10) if std != 0 else array


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

    # init
    v_s_values = []
    a_s_values = []

    for run_i in tqdm(range(100)):
        state, _ = env.reset()

        for step in range(100):
            encoded_state = encode(decode(state))
            q_values, _ = policy.model(torch.tensor(
                [encoded_state], dtype=torch.long))

            q_values = q_values.squeeze(0)
            v_s = q_values.mean().item()
            a_s = (q_values - v_s).detach().numpy()

            v_s_values.append(v_s)
            a_s_values.extend(a_s.tolist())

            # Perform the best action
            best_action = torch.argmax(q_values).item()
            state, _, terminated, truncated, _ = env.step(best_action)
            if terminated or truncated:
                break

    sns.kdeplot(a_s_values, fill=True, color="tomato", alpha=0.5)
    plt.title("Verteilung der Advantage-Funktion $A(s,a)$")
    plt.xlabel("Advantage-Funktion $A(s,a)$")
    plt.ylabel("Häufigkeit")
    plt.grid(True)
    plt.show()

    # NOTE: Undcomment for seeing the difference between normalized and non-normalized values

    # advantage_normalized = z_score_normalize(np.array(a_s_values))
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Two plots side by side

    # sns.kdeplot(a_s_values, fill=True, color="blue", alpha=0.5, ax=axes[0])
    # axes[0].set_title("Raw Advantage Function $A(s,a)$")
    # axes[0].set_xlabel("Advantage Function $A(s,a)$")
    # axes[0].set_ylabel("Density")
    # axes[0].grid(True)

    # sns.kdeplot(advantage_normalized, fill=True, color="tomato", alpha=0.5, ax=axes[1])
    # axes[1].set_title("Normalized Advantage Function $A'(s,a)$")
    # axes[1].set_xlabel("Normalized Advantage $A'(s,a)$")
    # axes[1].set_ylabel("Density")
    # axes[1].grid(True)

    # Show the comparison
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
