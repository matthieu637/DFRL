import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import numpy as np
from common.utils import RunningMeanStd
from .env_sumo_traffic_signal import TrafficSignal

# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# class Env(MultiAgentEnv):
class Env():
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param time_to_load_vehicles: (int) Number of simulation seconds ran before learning begins
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    """

    def __init__(self, normalize_inputs, T, doublereward, max_depart_delay=100000,
                 time_to_teleport=-1, time_to_load_vehicles=0, yellow_time=3, min_green=5, max_green=50):
        net_file = os.path.dirname(__file__)+'/../data/3x3Grid2lanes.net.xml'
        # route_file = os.path.dirname(__file__)+'/../data/routes14000.rou.xml'
        route_file = os.path.dirname(__file__) + '/../data/routes3x3.harder.rou.xml'
        use_gui = False
        time_to_load_vehicles = 200
        num_seconds = 5000 + time_to_load_vehicles
        delta_time = 10 #with delta_time at 5, the same action must be taken twice sometimes

        #two phase solution:
        # phases = [
        #     traci.trafficlight.Phase(35, "GGGgrrrrGGGgrrrr"),
        #     traci.trafficlight.Phase(2, "YYYYrrrrYYYYrrrr"),
        #     traci.trafficlight.Phase(35, "rrrrGGGgrrrrGGGg"),
        #     traci.trafficlight.Phase(2, "rrrrYYYYrrrrYYYY"),
        # ]

        # four phase solution:
        phases = [
            traci.trafficlight.Phase(300, "GGGrrrrrGGGrrrrr"),
            traci.trafficlight.Phase(3,  "yyyrrrrryyyrrrrr"),
            traci.trafficlight.Phase(300, "rrrGrrrrrrrGrrrr"),
            traci.trafficlight.Phase(3,  "rrryrrrrrrryrrrr"),
            traci.trafficlight.Phase(300, "rrrrGGGrrrrrGGGr"),
            traci.trafficlight.Phase(3,  "rrrryyyrrrrryyyr"),
            traci.trafficlight.Phase(300, "rrrrrrrGrrrrrrrG"),
            traci.trafficlight.Phase(3,  "rrrrrrryrrrrrrry")
        ]

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        self.doublereward = doublereward
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # start only to retrieve information

        self.ts_ids = traci.trafficlight.getIDList()
        self.lanes_per_ts = len(set(traci.trafficlight.getControlledLanes(self.ts_ids[0])))
        self.traffic_signals = dict()
        self.phases = phases
        self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
        self.vehicles = dict()
        self.last_measure = {}  # used to reward function remember last measure
        self.last_measure2 = []
        self.last_reward = [0. for _ in range(len(self.ts_ids))]
        self.sim_max_time = num_seconds
        self.time_to_load_vehicles = time_to_load_vehicles  # number of simulation seconds ran in reset() before learning starts
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.run = 0

        self.input_size = self.num_green_phases + 1 + 2 * self.lanes_per_ts
        self.n_actions = self.num_green_phases
        self.T = T
        self.max_steps = int((num_seconds-time_to_load_vehicles)/delta_time)
        self.n_agent = len(self.ts_ids)
        self.GAMMA = 0.99
        self.n_signal = 4
        self.n_episode = 2000
        self.nD = self.n_agent
        if doublereward:
            self.nD = self.n_agent*2
        self.max_u = None
        self.normalize_inputs = normalize_inputs
        self.compute_neighbors = False
        self.neighbors_size = 4 #max number of neighbor
        self.compute_neighbors_last = np.array([[1,3],[0,2],[1,5],[0,4,6],[1,3,5,7],[2,4,8],[3,7],[6,8],[5,7]])
        self.compute_neighbors_last_index = [list(range(len(self.compute_neighbors_last[i]))) for i in range(self.n_agent)]
        if normalize_inputs:
            self.obs_rms = [RunningMeanStd(shape=self.input_size) for _ in range(self.n_agent)]

        self.fileresults = open('learning.data', "w")
        traci.close()

    def __del__(self):
        self.fileresults.close()

    def toggle_compute_neighbors(self):
        pass

    def neighbors(self):
        return (self.compute_neighbors_last, self.compute_neighbors_last_index)

    def reset(self):
        self.metrics = []
        self.rinfo = np.array([0.] * 18)
        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '--max-depart-delay', str(self.max_depart_delay),
                     '--waiting-time-memory', '10000',
                     '--time-to-teleport', str(self.time_to_teleport),
                     '--random']
        if self.use_gui:
            sumo_cmd.append('--start')

        traci.start(sumo_cmd)

        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green, self.phases)
            self.last_measure[ts] = 0.0

        self.last_measure2=np.zeros(18)

        self.vehicles = dict()

        # Load vehicles
        for _ in range(self.time_to_load_vehicles):
            self._sumo_step()

        return self._get_obs()

    def _get_obs(self):
        """
        Return the current observation for each traffic signal
        """
        observations = []
        for ts in self.ts_ids:
            phase_id = [1 if self.traffic_signals[ts].phase//2 == i else 0 for i in range(self.num_green_phases)]  #one-hot encoding
            elapsed = self.traffic_signals[ts].time_on_phase / self.max_green
            density = self.traffic_signals[ts].get_lanes_density()
            queue = self.traffic_signals[ts].get_lanes_queue()
            observations.append(phase_id + [elapsed] + density + queue)

        if self.normalize_inputs:
            for i in range(self.n_agent):
                observations[i] = list(self.obs_rms[i].obs_filter(np.array(observations[i])))

        return observations

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()
    
    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()

        else:
            for i in range(self.n_agent):
                self.traffic_signals[self.ts_ids[i]].set_next_phase(action[i])

            for _ in range(self.yellow_time):
                self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update_phase()
            for _ in range(self.delta_time - self.yellow_time):
                self._sumo_step()

        # observe new state and reward
        observation = self._get_obs()
        reward, waiting_times = self._compute_rewards()
        done = self.sim_step > self.sim_max_time
        self.last_reward = reward
        self.rinfo += waiting_times
        if not self.doublereward:
            reward = np.array(reward).reshape(self.nD, -1).sum(axis=-1)

        return observation, reward, done

    def _compute_rewards(self):
        return self._waiting_time_reward_double2()

    # original
    def _waiting_time_reward(self):
        rewards = []
        for ts in self.ts_ids:
            ts_wait = sum(self.traffic_signals[ts].get_waiting_time())
            rewards.append(self.last_measure[ts] - ts_wait)
            self.last_measure[ts] = ts_wait
        return rewards

    def _waiting_time_reward_double(self):
        rewards = np.zeros(18)
        ts_wait = []
        for ts in self.ts_ids:
            ts_wait.append(self.traffic_signals[ts].get_waiting_time())
        ts_wait=np.array(ts_wait)
        newR = (ts_wait[:,0]+ts_wait[:,2])
        rewards[0::2] = self.last_measure2[:self.n_agent] - newR
        self.last_measure2[:self.n_agent] = newR

        newR = (ts_wait[:, 1] + ts_wait[:, 3])
        rewards[1::2] = self.last_measure2[self.n_agent:] - newR
        self.last_measure2[self.n_agent:] = newR

        return rewards
    
    def _waiting_time_reward_double2(self):
        rewards = np.zeros(18)
        ts_wait = []
        for ts in self.ts_ids:
            ts_wait.append(self.traffic_signals[ts].get_waiting_time())
        ts_wait=np.array(ts_wait)
        newR1 = (ts_wait[:,0]+ts_wait[:,2])
        rewards[0::2] = self.last_measure2[:self.n_agent] - newR1
        self.last_measure2[:self.n_agent] = newR1

        newR2 = (ts_wait[:, 1] + ts_wait[:, 3])
        rewards[1::2] = self.last_measure2[self.n_agent:] - newR2
        self.last_measure2[self.n_agent:] = newR2
        newRmerged = np.concatenate((newR1,newR2))

        return rewards, newRmerged
    
    def _waiting_time_reward_double3(self):
        rewards = np.zeros(18)
        ts_wait = []
        for ts in self.ts_ids:
            ts_wait.append(self.traffic_signals[ts].get_waiting_time())
        ts_wait=np.array(ts_wait)
        newR1 = (ts_wait[:,0]+ts_wait[:,2])
        rewards[0::2] = - newR1
        self.last_measure2[:self.n_agent] = newR1

        newR2 = (ts_wait[:, 1] + ts_wait[:, 3])
        rewards[1::2] = - newR2
        self.last_measure2[self.n_agent:] = newR2

        return rewards, -rewards
    

    def _compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'reward': np.array(self.last_reward).sum(),
            'total_stopped': sum([sum(self.traffic_signals[ts].get_stopped_vehicles_num()) for ts in self.ts_ids]),
            'total_wait_time': sum([self.last_measure[ts] for ts in self.ts_ids])
            #'total_wait_time': sum([sum(self.traffic_signals[ts].get_waiting_time()) for ts in self.ts_ids])
        }

    def _sumo_step(self):
        traci.simulationStep()

    def end_episode(self):
        self.fileresults.write(','.join(self.rinfo.flatten().astype('str')) + '\n')
        self.fileresults.flush()
        traci.close()

    def render(self):
        pass
