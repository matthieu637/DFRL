# Learning Fair Policies in Decentralized Cooperative Multi-Agent Reinforcement Learning [PDF](https://arxiv.org/abs/2012.09421)

We consider the problem of learning fair policies in (deep) cooperative multi-agent reinforcement learning (MARL). 
We formalize it in a principled way as the problem of optimizing a welfare function that explicitly encodes two important aspects of fairness: efficiency and equity. 
As a solution method, we propose a novel neural network architecture, which is composed of two sub-networks specifically designed for taking into account the two aspects of fairness. In experiments, we demonstrate the importance of the two sub-networks for fair optimization. 
Our overall approach is general as it can accommodate any (sub)differentiable welfare function.
Therefore, it is compatible with various notions of fairness that have been proposed in the literature (e.g., lexicographic maximin, generalized Gini social welfare function, proportional fairness).
Our solution method is generic and can be implemented in various MARL settings: centralized training and decentralized execution, or fully decentralized. 
Finally, we experimentally validate our approach in various domains and show that it can perform much better than previous methods.

## Installation
```
#Create a python virtual environment with python>=3.7
python3.7 -m venv py37-dfrl
#Activate the environment
. py37-dfrl/bin/activate
#Install the minimal requirements
pip install --upgrade pip
pip install tensorflow==1.15.4 keras==2.3.1 matplotlib

#For SUMO environments:
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

#If not done by your system after SUMO installation
#Define the SUMO_HOME environment variable by creating the file /etc/profile.d/sumo.sh or in your ~/.bashrc:
export SUMO_HOME=/usr/share/sumo

#For Iroko environment:
#Clone https://github.com/matthieu637/iroko/
#Follow the script https://github.com/matthieu637/iroko/blob/master/install.sh

```

## Training


```
#SOTO with alpha fairness in the centralized learning with decentralized execution scenario:
python SOTO-ALF-CLDE.py 

#SOTO with alpha fairness in the centralized learning with decentralized execution scenario:
python SOTO-ALF-CLDE.py 

#SOTO with GGF in the fully decentralized scenario:
python SOTO-ALF-FD.py 

#To change the environment, edit config.ini
#to reproduce our experiments, check the hyperparameters directory and replace config.ini
```

## Cite

If you make use of this code, please cite:

```
@misc{zimmer2021learning,
      title={Learning Fair Policies in Decentralized Cooperative Multi-Agent Reinforcement Learning}, 
      author={Matthieu Zimmer and Claire Glanois and Umer Siddique and Paul Weng},
      year={2021},
      eprint={2012.09421},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
``` 
