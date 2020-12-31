import os
import sys

import atexit
import ctypes
import logging
from multiprocessing import RawArray, RawValue
import numpy as np

from common.utils import RunningMeanStd
import iroko.utils as dc_utils
from iroko.control.iroko_bw_control import BandwidthController
from iroko.iroko_sampler import StatsSampler
from iroko.iroko_traffic import TrafficGen
from iroko.iroko_state import StateManager
from iroko.utils import TopoFactory
from iroko.topos.network_manager import NetworkManager

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONF = {
    # Input folder of the traffic matrix.
    "input_dir": f"{FILE_DIR}/../data/inputs/",
    # Which traffic matrix to run. Defaults to the first item in the list.
    "tf_index": 0,
    # Output folder for the measurements during trial runs.
    "output_dir": ".",
    # When to take state samples. Defaults to taking a sample at every step.
    "sample_delta": 1,
    # Use the simplest topology for tests.
    "topo": "fattree",
    # Which agent to use for traffic management. By default this is TCP.
    "agent": "tcp",
    # Which transport protocol to use. Defaults to the common TCP.
    "transport": "udp",
    # If we have multiple environments, we need to assign unique ids
    "parallel_envs": False,
    # Topology specific configuration (traffic pattern, number of hosts)
    "topo_conf": {"max_capacity": 100 * 1e6},
    # The network features supported by this environment
    "stats_dict": {"backlog": 0, "olimit": 1,
                   "drops": 2, "bw_rx": 3, "bw_tx": 4},
    # Specifies which variables represent the state of the environment:
    # Eligible variables are drawn from stats_dict
    # To measure the deltas between steps, prepend "d_" in front of a state.
    # For example: "d_backlog"
    "state_model": ["backlog"],
    # Add the flow matrix to state?
    "collect_flows": False,
    # Specifies which variables represent the state of the environment:
    # Eligible variables:
    # "action", "queue","std_dev", "joint_queue", "fair_queue"
    # "reward_model": ["joint_queue"],
    "reward_model": ["joint_queue_decomposed"],

}


def squash_action(action, action_min, action_max):
    action_diff = (action_max - action_min)
    return (np.tanh(action) + 1.0) / 2.0 * action_diff + action_min


def clip_action(action, action_min, action_max):
    """ Truncates the entries in action to the range defined between
    action_min and action_max. """
    return np.clip(action, action_min, action_max)


def sigmoid(action, derivative=False):
    sigm = 1. / (1. + np.exp(-action))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def clean():
    print("Removing all traces of Mininet")
    os.system('mn -c')
    os.system("killall -9 goben")
    os.system("killall -9 node_control")
    os.system("systemctl stop ovsdb-server.service")
    os.system("systemctl stop ovs-vswitchd.service")

class Env:

    def __init__(self, normalize_inputs, T, gamma, more_obs, average_rewards):
        if os.geteuid() != 0:
            exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")
        clean()
        os.system("systemctl start ovsdb-server.service")
        os.system("systemctl start ovs-vswitchd.service")

        self.T = T
        self.average_rewards = average_rewards
        self.normalize_inputs = normalize_inputs
        self.GAMMA = gamma
        self.max_steps = 10000000
        self.n_agent = 16
        self.nD = self.n_agent
        self.n_signal = 4
        self.n_episode = 300 #for around 3M transitions
        self.max_u = None
        self.input_size = 96
        self.n_actions = 1 #number of dim per agent

        self.conf = DEFAULT_CONF
        # self.conf.update(conf)

        if more_obs:
            self.conf.update({"state_model": ["backlog", "d_backlog", "olimit", "drops"]})
            self.input_size = 336

        # Init one-to-one mapped variables
        self.net_man = None
        self.state_man = None
        self.traffic_gen = None
        self.bw_ctrl = None
        self.sampler = None
        self.input_file = None
        self.terminated = False
        self.reward = RawValue('d', 0)

        # set the id of this environment
        self.short_id = dc_utils.generate_id()
        if self.conf["parallel_envs"]:
            self.conf["topo_conf"]["id"] = self.short_id
        # initialize the topology
        self.topo = TopoFactory.create(self.conf["topo"],
                                       self.conf["topo_conf"])
        # Save the configuration we have, id does not matter here
        dc_utils.dump_json(path=self.conf["output_dir"],
                           name="env_config", data=self.conf)
        dc_utils.dump_json(path=self.conf["output_dir"],
                           name="topo_config", data=self.topo.conf)
        # set the dimensions of the state matrix
        self._set_gym_matrices()
        # Set the active traffic matrix
        self._set_traffic_matrix(
            self.conf["tf_index"], self.conf["input_dir"], self.topo)

        # each unique id has its own sub folder
        if self.conf["parallel_envs"]:
            self.conf["output_dir"] += f"/{self.short_id}"
        # check if the directory we are going to work with exists
        dc_utils.check_dir(self.conf["output_dir"])

        # handle unexpected exits scenarios gracefully
        atexit.register(self.close)

        self.compute_neighbors = False
        self.neighbors_size = 4  # max number of neighbor
        self.compute_neighbors_last = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2],
                                                [5, 6, 7], [4, 6, 7], [4, 5, 7], [4, 5, 6],
                                                [9, 10, 11], [8, 10, 11], [8, 9, 11], [8, 9, 10],
                                                [13, 14, 15], [12, 14, 15], [12, 13, 15], [12, 13, 14]])
        self.compute_neighbors_last_index = [list(range(len(self.compute_neighbors_last[i]))) for i in range(self.n_agent)]
        if normalize_inputs:
            self.obs_rms = RunningMeanStd(shape=self.input_size)

        self.fileresults = open('learning.data', "w")

    def _set_gym_matrices(self):
        # set the action space
        num_actions = self.topo.get_num_hosts()
        min_bw = 10000.0 / float(self.topo.conf["max_capacity"])
        self.action_min = np.empty(num_actions)
        self.action_min.fill(min_bw)
        self.action_max = np.empty(num_actions)
        self.action_max.fill(1.0)
        #        self.action_space = spaces.Box(
        #            low=action_min, high=action_max, dtype=np.float32)
        # Initialize the action arrays shared with the control manager
        # Qdisc do not go beyond uint32 rate limit which is about 4Gbps
        tx_rate = RawArray(ctypes.c_uint32, num_actions)
        self.tx_rate = dc_utils.shmem_to_nparray(tx_rate, np.float32)
        active_rate = RawArray(ctypes.c_uint32, num_actions)
        self.active_rate = dc_utils.shmem_to_nparray(active_rate, np.float32)
        log.info("%s Setting action space", (self.short_id))
        log.info("from %s", self.action_min)
        log.info("to %s", self.action_max)

        # set the observation space
        num_ports = self.topo.get_num_sw_ports()
        num_features = len(self.conf["state_model"])
        if self.conf["collect_flows"]:
            num_features += num_actions * 2
        obs_min = np.empty(num_ports * num_features + num_actions)
        obs_min.fill(-np.inf)
        obs_max = np.empty(num_ports * num_features + num_actions)
        obs_max.fill(np.inf)
        #        self.observation_space = spaces.Box(
        #            low=obs_min, high=obs_max, dtype=np.float64)

    def __del__(self):
        self.fileresults.close()
        clean()

    def toggle_compute_neighbors(self):
        pass

    def neighbors(self):
        return (self.compute_neighbors_last, self.compute_neighbors_last_index)

    def _set_traffic_matrix(self, index, input_dir, topo):
        traffic_file = topo.get_traffic_pattern(index)
        self.input_file = f"{input_dir}/{topo.get_name()}/{traffic_file}"

    def _start_managers(self):
        # actually generate a topology if it does not exist yet
        if not self.net_man:
            log.info("%s Starting network manager...", self.short_id)
            self.net_man = NetworkManager(self.topo, self.conf["agent"].lower())
        # in a similar way start a traffic generator
        if not self.traffic_gen:
            log.info("%s Starting traffic generator...", self.short_id)
            self.traffic_gen = TrafficGen(self.net_man, self.conf["transport"], self.conf["output_dir"])
        # Init the state manager
        if not self.state_man:
            self.state_man = StateManager(self.conf,
                                          self.net_man,
                                          self.conf["stats_dict"])
        # Init the state sampler
        # if not self.sampler:
        #     stats = self.state_man.get_stats()
        #     self.sampler = StatsSampler(stats, self.tx_rate,
        #                                 self.reward, self.conf["output_dir"])
        #     self.sampler.start()

        # the bandwidth controller is reinitialized with every new network
        if not self.bw_ctrl:
            host_map = self.net_man.host_ctrl_map
            self.bw_ctrl = BandwidthController(
                host_map, self.tx_rate, self.active_rate, self.topo.max_bps)
            self.bw_ctrl.start()

    def _start_env(self):
        log.info("%s Starting environment...", self.short_id)
        # Launch all managers (if they are not active already)
        # This lazy initialization ensures that the environment object can be
        # created without initializing the virtual network
        self._start_managers()
        # Finally, start the traffic
        self.traffic_gen.start(self.input_file)

    def _stop_env(self):
        log.info("%s Stopping environment...", self.short_id)
        if self.traffic_gen:
            log.info("%s Stopping traffic", self.short_id)
            self.traffic_gen.stop()
        log.info("%s Done with stopping.", self.short_id)

    def reset(self):
        self.rinfo = np.array([0.] * self.n_agent)
        self.step_count = 0
        self._stop_env()
        self._start_env()
        return self._get_obs()

    def _get_obs(self):
        observations = self.state_man.observe()

        # Retrieve the bandwidth enforced by bandwidth control
        observations.extend(self.active_rate)

        observations = np.array(observations)
        if self.normalize_inputs:
            observations = self.obs_rms.obs_filter(np.array(observations))
        observations = list(observations)

        observations = [observations] * self.n_agent

        return observations

    def close(self):
        if self.terminated:
            return
        self.terminated = True
        log.info("%s Closing environment...", self.short_id)
        if self.state_man:
            log.info("%s Stopping all state collectors...", self.short_id)
            self.state_man.close()
            self.state_man = None
        if self.bw_ctrl:
            log.info("%s Shutting down bandwidth control...", self.short_id)
            self.bw_ctrl.close()
            self.bw_ctrl = None
        if self.sampler:
            log.info("%s Shutting down data sampling.", self.short_id)
            self.sampler.close()
            self.sampler = None
        if self.traffic_gen:
            log.info("%s Shutting down generators...", self.short_id)
            self.traffic_gen.close()
            self.traffic_gen = None
        if self.net_man:
            log.info("%s Stopping network.", self.short_id)
            self.net_man.stop_network()
            self.net_man = None
        log.info("%s Done with destroying myself.", self.short_id)

    def compute_rewards(self, action):
        return self.state_man.get_reward(action)

    def step(self, action):
        # Assume action is in [0 ; 1]
        action = np.array(action)[:, 0]
        action = self.action_min + action * (self.action_max - self.action_min)

        # Truncate actions to legal values
        # action = np.clip(action, self.action_min, self.action_max)

        # Retrieve observation and reward
        obs = self._get_obs()
        rewards = self.compute_rewards(action)
        self.reward.value = rewards.sum()
        self.rinfo += rewards
        self.step_count += 1

        # Update the array with the bandwidth control
        self.tx_rate[:] = action
        # The environment is finished when the traffic generators have stopped
        done = not self.traffic_gen.check_if_traffic_alive()
        return obs, rewards, done

    def _handle_interrupt(self, signum, frame):
        log.warning("%s \nEnvironment: Caught interrupt", self.short_id)
        atexit.unregister(self.close())
        self.close()
        sys.exit(1)

    def end_episode(self):
        if self.average_rewards:
            self.rinfo = self.rinfo / float(self.step_count)
        self.fileresults.write(','.join(self.rinfo.flatten().astype('str')) + '\n')
        self.fileresults.flush()

    def render(self):
        pass
