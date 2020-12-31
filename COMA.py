import numpy as np
from keras.utils import to_categorical
import copy
from common.utils import default_config, make_env, eligibility_traces, discount_rewards
from common.ppo_independant import PPOPolicyNetwork
from common.ppo_centralizedq import ValueNetwork

render = False
normalize_inputs = True

config = default_config()
env = make_env(config, normalize_inputs)
LAMBDA = float(config['agent']['lambda'])
lr_actor = float(config['agent']['lr_actor'])

n_agent = env.n_agent
T = env.T
GAMMA = env.GAMMA
n_episode = env.n_episode
max_steps = env.max_steps
n_actions = env.n_actions

i_episode = 0
qinput = env.input_size * n_agent + env.n_agent + n_actions * (n_agent-1)
meta_Q = ValueNetwork(num_features=qinput, hidden_size=128, num_output=n_actions, learning_rate=0.001)
meta_Pi = PPOPolicyNetwork(num_features=env.input_size + env.n_agent, num_actions=n_actions, layer_size=256, epsilon=0.1, learning_rate=lr_actor)

while i_episode < n_episode:
    i_episode += 1

    avg = [0] * n_agent

    ep_actions = [[] for _ in range(n_agent)]
    ep_pactions = [[] for _ in range(n_agent)]
    ep_rewards = [[] for _ in range(n_agent)]
    ep_states = [[] for _ in range(n_agent)]

    score = 0
    steps = 0
    su = [0.] * n_agent
    su = np.array(su)

    obs = env.reset()

    done = False
    while steps < max_steps and not done:
        steps += 1
        action = []
        for i in range(n_agent):
            h = copy.deepcopy(obs[i])
            h2 = copy.deepcopy(h)
            h.extend(to_categorical(i, n_agent))
            p = meta_Pi.get_dist(np.array([h]))[0]
            ep_pactions[i].append(p)
            action.append(np.random.choice(range(n_actions), p=p))
            ep_states[i].append(h2)
            ep_actions[i].append(to_categorical(action[i], n_actions))

        obs, rewards, done = env.step(action)

        su += np.array(rewards)
        score += sum(rewards)

        for i in range(n_agent):
            ep_rewards[i].append(rewards[i])

        if steps % T == 0:
            meta_state = np.array(ep_states).transpose(1, 0, 2).reshape(T, -1)
            meta_action = np.array(ep_actions).transpose(1, 0, 2).reshape(T, -1)
            mobs = np.array(obs).reshape(-1)
            meta_rewards = np.array(ep_rewards, dtype=np.float_).sum(axis=0)

            next_action = []
            for i in range(n_agent):
                h = copy.deepcopy(obs[i])
                h.extend(to_categorical(i, n_agent))
                p = meta_Pi.get_dist(np.array([h]))[0]
                next_action.append(to_categorical(np.random.choice(range(n_actions), p=p), n_actions))
            next_action = np.array(next_action)

            targets = []
            amstate = []
            allqsa = []
            for i in range(n_agent):
                ep_actions[i] = np.array(ep_actions[i])
                ep_states[i] = np.array(ep_states[i])

                rmmyindex = [j for j in range(n_agent) if i != j]
                others_action = np.array(ep_actions).transpose(1, 0, 2)[:, rmmyindex, :].reshape(T, -1)
                meta_state_s = np.concatenate((meta_state, to_categorical([i] * T, n_agent), others_action), axis=1)
                amstate.append(meta_state_s)
                qsall = meta_Q.get(meta_state_s)
                allqsa.append(qsall)
                qsa = (qsall * ep_actions[i]).sum(axis=-1)

                nothers_action = next_action[rmmyindex, :].reshape(-1)
                next_meta_state_s = np.concatenate((mobs, to_categorical(i, n_agent), nothers_action))
                next_qsall = meta_Q.get([next_meta_state_s])
                next_qsa = (next_qsall * next_action[i]).sum(axis=-1)

                ltarget = eligibility_traces(meta_rewards, qsa, next_qsa, GAMMA, LAMBDA)
                targets.append(ltarget)

            targets = np.array(targets).transpose()
            amstate = np.array(amstate)
            s, a, t = ([], [], [])
            for i in range(n_agent):
                s.append(amstate[i])
                a.append(ep_actions[i])
                t.append(targets[:, i])
            s = np.array(s).reshape((T*n_agent, -1))
            a = np.array(a).reshape((T*n_agent, -1))
            t = np.array(t).reshape((T*n_agent, -1))
            meta_Q.update(s, a, t[:, 0])

            #compute counterfactual
            allqsa = np.array(allqsa)
            ep_pactions = np.array(ep_pactions)
            baseline = (allqsa * ep_pactions).sum(axis=-1).transpose()
            ep_advantages = targets - baseline
            ep_advantages = (ep_advantages - np.mean(ep_advantages)) / (np.std(ep_advantages) + 0.0000000001)

            s, a, t = ([], [], [])
            for i in range(n_agent):
                s.append(np.concatenate((ep_states[i], to_categorical([i] * T, n_agent)), axis=-1))
                a.append(ep_actions[i])
                t.append(ep_advantages[:, i])
            s = np.array(s).reshape((T*n_agent, -1))
            a = np.array(a).reshape((T*n_agent, -1))
            adv = np.array(t).reshape((T*n_agent, -1))

            meta_Pi.update(s, a, adv[:, 0])

            ep_actions = [[] for _ in range(n_agent)]
            ep_pactions = [[] for _ in range(n_agent)]
            ep_rewards = [[] for _ in range(n_agent)]
            ep_states = [[] for _ in range(n_agent)]

        if render:
            env.render()

    print(i_episode)
    print(score / max_steps)
    print(su)
    uti = np.array(su) / max_steps

    print(env.rinfo.flatten())
    env.end_episode()
