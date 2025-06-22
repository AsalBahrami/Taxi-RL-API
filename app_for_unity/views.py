import os
import json
import io
from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import torch
import numpy as np
import gymnasium as gym
import tianshou as ts
import time

from .bl_taxi_test import BLDuelingDQN, decode, encode

# Environment initialization
env = gym.make('Taxi-v3', render_mode='ansi')
state, _ = env.reset()
acc_reward = 0
terminated, truncated, counter = False, False, 0


global_max_i_s = -float('inf')
global_min_i_s = float('inf')
i_s_values = np.zeros((5, 5))
max_q_values = np.zeros((5, 5))
min_q_values = np.zeros((5, 5))
# global_cached_i_values_normalized = None
episode_start_time = None



@csrf_exempt
def run_taxi_view(request):
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

    global state, acc_reward, terminated, truncated, counter, episode_start_time
    # global  global_max_i_s, global_min_i_s, i_s_values
    global global_max_i_s, global_min_i_s, i_s_values, global_cached_i_values_normalized


    if request.method == 'GET':
        state, _ = env.reset()
        decoded_state = list(decode(state))
        episode_start_time = time.time()

        i_s_values = np.zeros((5, 5))
        for x in range(5):
            for y in range(5):
                s = decode(state)
                s[0], s[1] = y, x
                s_encoded = encode(s)
                q_vals, _ = policy.model(torch.tensor([s_encoded]))
                i_s_values[x, y] = q_vals.max().item() - q_vals.min().item()
        
        global_min_i_s = np.min(i_s_values)
        global_max_i_s = np.max(i_s_values)
        
        global_cached_i_values_normalized = (i_s_values - global_min_i_s) / (global_max_i_s - global_min_i_s + 1e-10)
        
        initial_data = {
            'state': state,
            'decoded_state': decoded_state,
        }
        return JsonResponse(initial_data)

    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_action = int(data.get('action', -1))
            if user_action not in range(env.action_space.n):
                return JsonResponse({'error': 'Invalid action'}, status=400)

            # Step environment with the user's action
            state, reward, terminated, truncated, info = env.step(user_action)
            elapsed_time = None
            if terminated or truncated:
                if episode_start_time:
                    elapsed_time = time.time() - episode_start_time
                    print(f"Episode finished in {elapsed_time:.2f} seconds.")
                    episode_start_time = None  # Reset timer

            decoded_state = list(decode(state))
            taxi_row, taxi_col, passenger_loc, target_loc = decoded_state
            print(f"[Backend Debug] Taxi at (row={taxi_row}, col={taxi_col})")
            print(f"[Backend Debug] Passenger location index: {passenger_loc}")
            print(f"[Backend Debug] Target (lamp) location index: {target_loc}")

            q_values, _ = policy.model([state])

            best_action = torch.argmax(q_values).item()
            action_names = ["down", "up", "left", "right",
                            "pickup passenger", "drop off passenger"]

            best_action_name = (
                action_names[best_action]
                if 0 <= best_action < len(action_names)
                else "unknown"
            )

            function_type = os.getenv('FUNCTION_TYPE', 'value')


            if function_type == "i_function":
                if global_cached_i_values_normalized is None:
                    return JsonResponse({'error': 'I-Function not initialized'}, status=500)
                
                max_q_values_normalized = global_cached_i_values_normalized.flatten().tolist()
                # Uncommment to debug
                # print(f"Re-using cached max_q_values normalized: {max_q_values_normalized}")
            elif function_type == "q_value":

                action_dim = action_shape  # e.g. 6 for Taxi‐v3
                q_values_grid = np.zeros((5, 5, action_dim), dtype=np.float32)

                for x in range(5):
                    for y in range(5):
                        modified_state = decode(state)
                        modified_state[0] = y
                        modified_state[1] = x
                        encoded_state = encode(modified_state)
                        q_values, _ = policy.model(
                            torch.tensor([encoded_state]))
                        q_values_grid[x, y, :] = q_values.detach().numpy()
                # shape: (5,5,4)
                q_move = q_values_grid[:, :, :4] 

                q_min, q_max = np.min(q_move), np.max(q_move)
                q_norm = (q_move - q_min) / (q_max - q_min + 1e-10)

                q_values_normalized = q_norm.flatten().tolist()

            elif function_type == "value":
                max_q_values = np.zeros((5, 5))
                for x in range(5):
                    for y in range(5):
                        modified_state = decode(state)
                        modified_state[0] = y
                        modified_state[1] = x
                        encoded_state = encode(modified_state)
                        q_value, _ = policy.model(
                            torch.tensor([encoded_state]))
                        max_q_values[x, y] = q_value.max().item()

                max_q_values_normalized = (max_q_values - np.min(max_q_values)) / (
                    np.max(max_q_values) - np.min(max_q_values))
                max_q_values_normalized = max_q_values_normalized.flatten().tolist()

            elif function_type == "advantage":
                q_values_grid = np.zeros((5, 5, 6))
                for x in range(5):
                    for y in range(5):
                        s = decode(state)
                        s[0], s[1] = y, x
                        s_encoded = encode(s)
                        q_vals, _ = policy.model(torch.tensor([s_encoded]))
                        q_values_grid[x, y, :] = q_vals.detach().numpy()

                # 1. Compute Advantage normally: A(s,a) = Q(s,a) - mean_a Q(s,a)
                advantage = q_values_grid - q_values_grid.mean(axis=2, keepdims=True)
                advantage = advantage[:, :, :4]  # Use only 4 movement actions

                # 2. Normalize across the entire grid (global)
                adv_min = np.min(advantage)
                adv_max = np.max(advantage)
                adv_norm = (advantage - adv_min) / (adv_max - adv_min + 1e-10)

                # 3. Flatten for sending
                advantage_grid_flat = adv_norm.flatten().tolist()
                
            response_data = {
                'state': state,
                'decoded_state': decoded_state,
                'reward': reward,
                'done': terminated or truncated,
                'best_action': best_action,
                'best_action_name': best_action_name,
                'max_q_values_normalized': max_q_values_normalized if function_type in ["value", "i_function"] else [],
                'advantage_grid_flat': advantage_grid_flat if function_type == "advantage" else [],
                'q_values_normalized': q_values_normalized if function_type == "q_value" else [],
                'function_type': function_type,
                'episode_duration': elapsed_time,
                # 'advantage_raw': q_values_grid[:, :, :4].tolist(),
            }
            return JsonResponse(response_data)

        except json.JSONDecodeError as e:
            return HttpResponseBadRequest(f'json decode error: {str(e)}')
        except KeyError as e:
            return HttpResponseBadRequest(f'json key error: {str(e)}')
        except Exception as e:
            return HttpResponseBadRequest(f'badrequest: {str(e)}')
