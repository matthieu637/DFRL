import numpy as np
from keras.utils import to_categorical
import copy
from common.utils import eligibility_traces, default_config, make_env, RunningMeanStd, str2bool, discount_rewards
from common.ppo_independant import PPOPolicyNetwork, ValueNetwork

render = False
normalize_inputs = True

config = default_config()
LAMBDA = float(config['agent']['lambda'])
lr_actor = float(config['agent']['lr_actor'])
meta_skip_etrace = str2bool(config['agent']['meta_skip_etrace'])
env = make_env(config, normalize_inputs)

n_agent = env.n_agent
T = env.T
GAMMA = env.GAMMA
n_episode = env.n_episode
max_steps = env.max_steps
n_actions = env.n_actions
n_signal = env.n_signal
max_u = env.max_u

i_episode = 0
meta_Pi = []
meta_V = []
for i in range(n_agent):
    meta_Pi.append(PPOPolicyNetwork(num_features=env.input_size + 2, num_actions=n_signal, layer_size=128, epsilon=0.1, learning_rate=lr_actor))
    meta_V.append(ValueNetwork(num_features=env.input_size + 2, hidden_size=128, learning_rate=0.001))

Pi = [[] for _ in range(n_agent)]
V = [[] for _ in range(n_agent)]
for i in range(n_agent):
    for j in range(n_signal):
        Pi[i].append(PPOPolicyNetwork(num_features=env.input_size, num_actions=n_actions, layer_size=256, epsilon=0.1, learning_rate=lr_actor))
        V[i].append(ValueNetwork(num_features=env.input_size, hidden_size=256, learning_rate=0.001))

if normalize_inputs:
    meta_obs_rms = [RunningMeanStd(shape=2) for _ in range(n_agent)]

while i_episode < n_episode:
    i_episode += 1

    avg = [0] * n_agent
    u_bar = [0] * n_agent
    utili = [0] * n_agent
    u = [[] for _ in range(n_agent)]

    ep_actions = [[] for _ in range(n_agent)]
    ep_rewards = [[] for _ in range(n_agent)]
    ep_states = [[] for _ in range(n_agent)]

    meta_z = [[] for _ in range(n_agent)]
    meta_rewards = [[] for _ in range(n_agent)]
    meta_states = [[] for _ in range(n_agent)]

    signal = [0] * n_agent
    rat = [0.0] * n_agent

    score = 0
    steps = 0
    su = [0.] * n_agent
    su = np.array(su)

    obs = env.reset()

    done = False
    while steps < max_steps and not done:

        if steps % T == 0:
            for i in range(n_agent):
                h = copy.deepcopy(obs[i])
                h.append(rat[i])
                h.append(utili[i])
                if normalize_inputs:
                    h[-2:] = list(meta_obs_rms[i].obs_filter(np.array(h)[-2:]))
                p_z = meta_Pi[i].get_dist(np.array([h]))[0]
                z = np.random.choice(range(n_signal), p=p_z)
                signal[i] = z
                meta_z[i].append(to_categorical(z, n_signal))
                meta_states[i].append(h)

        steps += 1
        action = []
        for i in range(n_agent):
            h = copy.deepcopy(obs[i])
            p = Pi[i][signal[i]].get_dist(np.array([h]))[0]
            action.append(np.random.choice(range(n_actions), p=p))
            ep_states[i].append(h)
            ep_actions[i].append(to_categorical(action[i], n_actions))

        obs, rewards, done = env.step(action)

        su += np.array(rewards)
        score += sum(rewards)

        for i in range(n_agent):
            u[i].append(rewards[i])
            u_bar[i] = sum(u[i]) / len(u[i])
        '''
		avg=copy.deepcopy(u_bar)
		for j in range(10):
			for i in range(n_agent):
				s=0
				for k in range(3):
					m=np.random.randint(0,n_agent)
					s+=avg[m]
				avg[i]=(avg[i]*0.02+(avg[i]+s)/(3+1)*0.98)+(np.random.rand()-0.5)*0.0001
		'''
        for i in range(n_agent):
            avg[i] = sum(u_bar) / len(u_bar)
            if avg[i] != 0:
                rat[i] = (u_bar[i] - avg[i]) / avg[i]
            else:
                rat[i] = 0
            # print(avg[i])#might help to define max_u
            if max_u != None:
                utili[i] = min(1, avg[i] / max_u)
            else:
                utili[i] = avg[i]

        for i in range(n_agent):
            if signal[i] == 0:
                ep_rewards[i].append(rewards[i])
            else:
                h = copy.deepcopy(obs[i])
                h.append(rat[i])
                h.append(utili[i])
                if normalize_inputs:
                    h[-2:] = list(meta_obs_rms[i].obs_filter(np.array(h)[-2:]))
                p_z = meta_Pi[i].get_dist(np.array([h]))[0]
                r_p = p_z[signal[i]]
                ep_rewards[i].append(r_p)

        if steps % T == 0:
            for i in range(n_agent):
                meta_rewards[i].append(utili[i] / (0.1 + abs(rat[i])))
                ep_actions[i] = np.array(ep_actions[i])
                ep_rewards[i] = np.array(ep_rewards[i], dtype=np.float_)
                ep_states[i] = np.array(ep_states[i])
                if LAMBDA < -0.1:
                    targets = discount_rewards(ep_rewards[i], GAMMA)
                    V[i][signal[i]].update(ep_states[i], targets)
                    vs = V[i][signal[i]].get(ep_states[i])
                else:
                    vs = V[i][signal[i]].get(ep_states[i])
                    targets = eligibility_traces(ep_rewards[i], vs, V[i][signal[i]].get(copy.deepcopy([obs[i]])), GAMMA, LAMBDA)
                    V[i][signal[i]].update(ep_states[i], targets)
                ep_advantages = targets - vs
                ep_advantages = (ep_advantages - np.mean(ep_advantages)) / (np.std(ep_advantages) + 0.0000000001)
                Pi[i][signal[i]].update(ep_states[i], ep_actions[i], ep_advantages)

            ep_actions = [[] for _ in range(n_agent)]
            ep_rewards = [[] for _ in range(n_agent)]
            ep_states = [[] for _ in range(n_agent)]

        if render:
            env.render()

    for i in range(n_agent):
        if len(meta_rewards[i]) == 0:
            continue
        meta_z[i] = np.array(meta_z[i])
        meta_rewards[i] = np.array(meta_rewards[i])
        meta_states[i] = np.array(meta_states[i])
        if done:
            meta_states[i] = meta_states[i][:len(meta_rewards[i]), :]
            meta_z[i] = meta_z[i][:len(meta_rewards[i]), :]
        meta_vs = meta_V[i].get(meta_states[i])
        if meta_skip_etrace:
            meta_targets = meta_rewards[i]
        else:
            h = copy.deepcopy(obs[i])
            h.append(rat[i])
            h.append(utili[i])
            meta_targets = eligibility_traces(meta_rewards[i], meta_vs, meta_V[i].get([h]), GAMMA, LAMBDA)
        meta_V[i].update(meta_states[i], meta_targets)
        meta_advantages = meta_targets - meta_vs
        meta_Pi[i].update(meta_states[i], meta_z[i], meta_advantages)
    print(i_episode)
    print(score / max_steps)
    print(su)
    uti = np.array(su) / max_steps

    print(env.rinfo.flatten())
    env.end_episode()
