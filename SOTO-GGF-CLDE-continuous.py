import numpy as np
from keras.utils import to_categorical
import copy
from common.utils import eligibility_traces, default_config, make_env, str2bool, get_omega, get_more_obs_com, discount_rewards
from common.ppo_independant import PPOPolicyNetwork, ValueNetwork, CPPOPolicyNetwork
from collections import deque

render = False
normalize_inputs = True

config = default_config()
LAMBDA = float(config['agent']['lambda'])
lr_actor = float(config['agent']['lr_actor'])
twophase_proportion = float(config['agent']['twophase_proportion'])
if_tanh = str2bool(config['agent']['if_tanh'])
stdev = float(config['agent']['stdev'])

env = make_env(config, normalize_inputs)
env.toggle_compute_neighbors()

n_agent = env.n_agent
T = env.T
GAMMA = env.GAMMA
n_episode = env.n_episode
max_steps = env.max_steps
n_actions = env.n_actions
omega = get_omega(config, n_agent)

i_episode = 0
gPi = []
Pi = []
gV = []
V = []
more_obs_size=env.neighbors_size+1
more_obs_size2 = n_agent

for i in range(n_agent):
    gPi.append(CPPOPolicyNetwork(num_features=env.input_size, if_tanh=if_tanh, num_actions=n_actions, layer_size=256, epsilon=0.1,
                               learning_rate=lr_actor, std=stdev))
    Pi.append(CPPOPolicyNetwork(num_features=env.input_size+more_obs_size+more_obs_size2+n_actions, if_tanh=if_tanh, num_actions=n_actions,
                                layer_size=64, epsilon=0.1, learning_rate=lr_actor, std=stdev))
    gV.append(ValueNetwork(num_features=env.input_size, hidden_size=256, learning_rate=0.001))
    V.append(ValueNetwork(num_features=env.input_size+more_obs_size+more_obs_size2+n_actions, hidden_size=256, learning_rate=0.001))

memory_ep_rewards = [deque() for _ in range(n_agent)]
average_jpi = np.zeros(n_agent)

while i_episode < n_episode:
    beta = max(1 - float(i_episode) / (twophase_proportion * float(n_episode)), 0.0)
    i_episode += 1

    memory_ep_rewards = [deque() for _ in range(n_agent)]
    average_jpi = np.zeros(n_agent)

    avg = [0.] * n_agent

    ep_actions = [[] for _ in range(n_agent)]
    ep_neglogproba = [[] for _ in range(n_agent)]
    ep_rewards = [[] for _ in range(n_agent)]
    ep_states  = [[] for _ in range(n_agent)]

    greedy = np.zeros(n_agent).astype(bool)
    for i in range(n_agent):
        greedyc = np.random.rand() <= beta
        greedy[i] = greedyc

    score = 0
    steps = 0
    su = [0.] * n_agent
    su = np.array(su)

    obs = env.reset()
    neighbors = env.neighbors()

    done = False
    while steps < max_steps and not done:
        steps += 1
        action = []
        for i in range(n_agent):
            h = copy.deepcopy(obs[i])
            if not greedy[i]:
                more_obs = gPi[i].get_mean(np.array([h]))[0]
                h.extend(more_obs)
                more_obs = get_more_obs_com(True, neighbors, average_jpi, i, more_obs_size)
                h.extend(more_obs)
                more_obs = average_jpi
                more_obs = (more_obs - np.mean(more_obs)) / (np.std(more_obs) + 0.0000000001)
                h.extend(more_obs)
                s, nlp = Pi[i].get_dist(np.array([h]))
            else:
                s, nlp = gPi[i].get_dist(np.array([h]))
            ep_states[i].append(h)
            action.append(s[0])
            ep_actions[i].append(s[0])
            ep_neglogproba[i].append(nlp[0])

        obs, rewards, done = env.step(action)
        neighbors = env.neighbors()

        su += np.array(rewards)
        score += sum(rewards)

        for i in range(n_agent):
            ep_rewards[i].append(rewards[i])
            memory_ep_rewards[i].append(rewards[i])
            average_jpi[i] += rewards[i]
            if len(memory_ep_rewards[i]) > max_steps * 5:
                average_jpi[i] -= memory_ep_rewards[i].popleft()

        if steps % T == 0:
            all_ep_advantages=[]
            for i in range(n_agent):
                ep_actions[i] = np.array(ep_actions[i])
                ep_neglogproba[i] = np.array(ep_neglogproba[i])
                ep_rewards[i] = np.array(ep_rewards[i], dtype=np.float_)
                ep_states[i] = np.array(ep_states[i])

                if LAMBDA < -0.1:
                    targets = discount_rewards(ep_rewards[i], GAMMA)
                    if not greedy[i]:
                        V[i].update(ep_states[i], targets)
                        vs = V[i].get(ep_states[i])
                    else:
                        gV[i].update(ep_states[i], targets)
                        vs = gV[i].get(ep_states[i])
                else:
                    next_s = copy.deepcopy(obs[i])
                    if not greedy[i]:
                        vs = V[i].get(ep_states[i])
                        more_obs = gPi[i].get_dist(np.array([obs[i]]))[0]
                        next_s.extend(more_obs)
                        more_obs = get_more_obs_com(True, neighbors, average_jpi, i, more_obs_size)
                        next_s.extend(more_obs)
                        more_obs = average_jpi
                        more_obs = (more_obs - np.mean(more_obs)) / (np.std(more_obs) + 0.0000000001)
                        next_s.extend(more_obs)
                        targets = eligibility_traces(ep_rewards[i], vs, V[i].get([next_s]), GAMMA, LAMBDA)
                        V[i].update(ep_states[i], targets)
                    else:
                        vs = gV[i].get(ep_states[i])
                        targets = eligibility_traces(ep_rewards[i], vs, gV[i].get([next_s]), GAMMA, LAMBDA)
                        gV[i].update(ep_states[i], targets)

                ep_advantages = targets - vs
                ep_advantages = (ep_advantages - np.mean(ep_advantages)) / (np.std(ep_advantages) + 0.0000000001)
                all_ep_advantages.append(ep_advantages)

            sorted_index = average_jpi.argsort()
            sorted_index = [np.where(sorted_index == i)[0][0] for i in range(n_agent)]
            all_ep_advantages = np.array(all_ep_advantages)
            all_ep_advantages_saved = all_ep_advantages
            all_ep_advantages = omega[sorted_index] @ all_ep_advantages
            for i in range(n_agent):
                if not greedy[i]:
                    Pi[i].update(ep_states[i], ep_actions[i], ep_neglogproba[i], all_ep_advantages)
                else:
                    gPi[i].update(ep_states[i], ep_actions[i], ep_neglogproba[i], all_ep_advantages_saved[i])

            ep_actions = [[] for _ in range(n_agent)]
            ep_neglogproba = [[] for _ in range(n_agent)]
            ep_rewards = [[] for _ in range(n_agent)]
            ep_states  = [[] for _ in range(n_agent)]

            greedy=np.zeros(n_agent).astype(bool)
            for i in range(n_agent):
                greedyc = np.random.rand() <= beta
                greedy[i] = greedyc

        if render:
            env.render()

    print(i_episode)
    print(score / max_steps)
    print(omega @ su[su.argsort()])
    print(su)

    print(env.rinfo.flatten())
    env.end_episode()

