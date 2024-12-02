# CSCE-642: Deep Reinforcement Learning

## Setup

SWIG is required for installing Box2D. It can be installed on Linux by running 
```bash
sudo apt-get install swig build-essential python-dev python3-dev
```
and on Mac by running
```bash
brew install swig
```
or on windows by following the instructions [here](https://open-box.readthedocs.io/en/latest/installation/install_swig.html).

For setting up the environment, we recommend using conda + pip or virtual env + pip. The Python environment required is 3.9.16 (version)

 Install the packages given by
```bash
pip install -r requirements.txt
```

The commands to run the 3 different mode are here:
1) DQN model
```bash
python run.py -s dqn -d MontezumaRevenge -e 1000
```

2) DQN + ICM model
```bash
python run.py -s dqn_icm -d MontezumaRevenge -e 1000 -t 2500 --icm_beta 0.2 --icm_eta 0.01
```
3) DQN + ICM temporally enhanced model
```bash
python run.py -s dqn_icm_temporal -d MontezumaRevenge -e 1000 -t 2500 --icm_beta 0.2 --icm_eta 0.01
```

