import os
import json
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
import torch
import numpy as np
from .bl_taxi_test import BLDuelingDQN, decode, encode
import gymnasium as gym
import tianshou as ts

# Umgebung initialisieren
env = gym.make('Taxi-v3', render_mode='ansi')
state, _ = env.reset()
acc_reward = 0
terminated, truncated, counter = False, False, 0


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

    global state, acc_reward, terminated, truncated, counter

    # Unity sendet eine GET-Anfrage
    if request.method == 'GET':
        state, _ = env.reset()
        decoded_state = list(decode(state))
        initial_data = {
            'state': state,
            'decoded_state': decoded_state,
        }
        return JsonResponse(initial_data)

    # Action von Unity zu Django gesendet
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_action = int(data.get('action', -1))
            if user_action not in range(env.action_space.n):
                return JsonResponse({'error': 'Invalid action'}, status=400)

            state, reward, terminated, truncated, info = env.step(user_action)
            decoded_state = list(decode(state))

            q_values, _ = policy.model([state])

            # Determine function type
            function_type = os.getenv('FUNCTION_TYPE', 'value')

            if function_type == "value":
                # calculate value function 
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

                max_q_values_normalized = (
                    max_q_values - np.min(max_q_values)) / (np.max(max_q_values) - np.min(max_q_values))
                max_q_values_normalized = max_q_values_normalized.flatten().tolist()

            # calculate advantage function I(s)
            elif function_type == "advantage":
                # Compute the advantage function I(s)
                max_q = q_values.max().item()
                min_q = q_values.min().item()
                i_s = max_q - min_q

                max_q_values_normalized = np.full((5, 5), i_s)
                max_q_values_normalized = max_q_values_normalized.flatten().tolist() 

            else:
                return JsonResponse({'error': 'Invalid function type'}, status=400)

            response_data = {
                'state': state,
                'decoded_state': decoded_state,
                'reward': reward,
                'max_q_values_normalized': max_q_values_normalized,
                # indicates which function get used
                'function_type': function_type
            }
            return JsonResponse(response_data)

        except json.JSONDecodeError as e:
            return HttpResponseBadRequest(f'json decode error: {str(e)}')
        except KeyError as e:
            return HttpResponseBadRequest(f'json key error: {str(e)}')
        except Exception as e:
            return HttpResponseBadRequest(f'badrequest: {str(e)}')
