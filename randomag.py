import numpy as np
from common.utils import default_config, make_env

render = False
normalize_inputs = True

config = default_config()
env = make_env(config, normalize_inputs)

n_agent = env.n_agent
nD = env.nD
n_episode = env.n_episode
max_steps = env.max_steps
n_actions = env.n_actions

i_episode = 0

while i_episode<n_episode:
	i_episode+=1

	score=0
	steps=0
	su=[0.]*nD
	su = np.array(su)

	obs = env.reset()

	done = False
	while steps<max_steps and not done:
		steps+=1
		action=[]
		for i in range(n_agent):
			action.append(np.random.choice(range(n_actions)))
		
		obs, rewards, done = env.step(action)

		su+=np.array(rewards)
		score += sum(rewards)

		# if steps % 100 == 0:
		# 	print(steps)

		if render:
			env.render()

	print(i_episode)
	print(score/max_steps)
	print(su)
	uti = np.array(su)/max_steps

	print(env.rinfo.flatten())
	env.end_episode()
