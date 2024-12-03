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

The commands to run the 3 different models are here:
1) DQN model
```bash
python run.py -s dqn -d MontezumaRevenge -e 1000
```
Side note for the DQN model you do have to modify the render mode in the run.py file. If running the baseline DQN then the render mode should be none. Otherwise the render mode should be rgb_array
```python
    elif domain == "MontezumaRevenge":
        if not render_mode:
            render_mode = 'rgb_array'
            # Keep it None for DQN only otherwise rgb_array
        return gym.make("ALE/MontezumaRevenge-v5", render_mode=None)
```

2) DQN + ICM model
```bash
python run.py -s dqn_icm -d MontezumaRevenge -e 1000 -t 2500 --icm_beta 0.2 --icm_eta 0.01
```
```python
    elif domain == "MontezumaRevenge":
        if not render_mode:
            render_mode = 'rgb_array'
            # Keep it None for DQN only otherwise rgb_array
        return gym.make("ALE/MontezumaRevenge-v5", render_mode=render_mode)
```

3) DQN + ICM temporally enhanced model
```bash
python run.py -s dqn_icm_temporal -d MontezumaRevenge -e 1000 -t 2500 --icm_beta 0.2 --icm_eta 0.01
```

## Acknowledgements
Base DQN code is based on the CSCE 642 in-class [codebase](https://github.com/Pi-Star-Lab/csce642-deepRL.git).

Base ICM code is based on Curiosity-driven Exploration by Self-supervised Prediction [codebase](https://github.com/pathak22/noreward-rl.git).