# Deep Reinforcement Learning Nanodegree at Udacity

This repository will contain my solutions to the three projects within the course. The projects are:
1. [Training an agent to collect yellow bananas while avoiding blue bananas.](#project-1-banana-collector)
2. [Training a robotic arm to reach target locations.](#project-2-reacher)
3. [Training a pair of agents to play tennis.](#project-3-tennis)

## Project 1: Banana Collector
### Project Details
For this project, when collecting yellow bananas the environment is providing a reward of +1 whereas collecting a blue banana results in a reward of -1. To conclude, the agent should collect as many yellow bananas as possible to maximize the reward.

The state space has **37 dimensions** and contains
* the agent's velocity,
* along with ray-based perception of objects around the agent's forward direction.

Given this information, the agent has to learn how to best select actions. **Four discrete actions** are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an **average score of +13 over 100 consecutive episodes**.

Note: The project environment is similar to, but not identical to the Banana Collector environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).

### Getting Started
The following instructions assume you are running on an Ubuntu 16.04 x64 distribution, with an [Anaconda](https://www.anaconda.com/download/#linux) installation. If you have installed Anaconda, please run following commands in order to execute the **Banana Collector** notebook:

Note: the python folder within this repository and the link to the pre-build Unity Environment for Linux are copied from the [Udacity DRLND course](https://github.com/udacity/deep-reinforcement-learning)
```bash
$ conda create --name drlnd python=3.6
$ source activate drlnd

$ git clone https://github.com/shelbyi/drlnd.git
$ cd drlnd/python
$ pip install .

$ python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
Next, download the pre-build Unity Environment for your Linux system.
<br>Linux visual: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
<br>Linux headless: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)
<br>Place the file in the **project1-banana-collector** folder within the repository and unzip it.

Now, you should be set up to run the notebook on your system. 

### Instructions
Before you run the notebook, be sure to change the kernel to **drlnd**. Execute cell after cell to get some information of the environment and to train the agent to collect yellow bananas. At the end, the neural network model and the **replay buffer** are stored to disk under **[path to the repo]/drlnd/project1-banana-collector/models**.

Start the notebook and execute the cells.
```
$ cd [path to the repo]/drlnd/project1-banana-collector/
$ jupyter-notebook banana-collector.ipynb
``` 

## Project 2: Reacher
### Project Details
For this project, the target of the agent is to control a single double-jointed arm that follows a goal location (which is a moving target) and keeps its hand within this location. The environment is providing a reward of +0.1. To conclude, the agent should keep its "hand" as long as possible within the target area to maximize the reward.

The state space has **33 dimensions** and corresponds to
* position
* rotation
* velocity
* angular velocities
of the arm.

Given this information, the agent has to learn how to best select actions. Each **continuous action** (consisting of a vector of four entries) corresponds to the torque applicable to two joints. Each entry in the action vector should be a number between -1 and +1.

The task is episodic, and in order to solve the environment, the agent must get an **average score of +30 over 100 consecutive episodes**.

Note: The project environment is similar to, but not identical to the Reacher environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

### Getting Started
The following instructions assume you are running on an Ubuntu 16.04 x64 distribution, with an [Anaconda](https://www.anaconda.com/download/#linux) installation. If you have installed Anaconda, please run following commands in order to execute the **Reacher** notebook:

Note: the python folder within this repository and the link to the pre-build Unity Environment for Linux are copied from the [Udacity DRLND course](https://github.com/udacity/deep-reinforcement-learning)
```bash
$ conda create --name drlnd python=3.6
$ source activate drlnd

$ git clone https://github.com/shelbyi/drlnd.git
$ cd drlnd/python
$ pip install .

$ python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
Next, download the pre-build Unity Environment for your Linux system.
<br>Linux visual: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
<br>Linux headless: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip)
<br>Place the file in the **project2-reacher** folder within the repository and unzip it.

Now, you should be set up to run the notebook on your system.

### Instructions
Before you run the notebook, be sure to change the kernel to **drlnd**. Execute cell after cell to get some information of the environment and to train the agent to keep its "hand" within the target location. At the end, the neural networks of the actor-critic model and the **replay buffer** are stored to disk under **[path to the repo]/drlnd/project2-reacher/models**.

Start the notebook and execute the cells.
```
$ cd [path to the repo]/drlnd/project2-reacher/
$ jupyter-notebook reacher.ipynb
``` 

## Project 3: Tennis
### Project Details
For this project, the target is to train two agents that will be able to control a racket to bounce a ball over a net. The environment is providing a reward of +0.1 if the agent was able to hit the ball over the net. It will get a reward of -0.01 if the agent let the ball fall on the ground or if it hit the ball out of bounce. To conclude, the agents should learn a strategy to cooperate and to keep the ball in the game as long as possible to maximize the reward.

The state space has **8 dimensions** and corresponds to 
* velocity
* position
of the ball and racket.

Given this information, the agent has to learn how to best control the racket and its position. Each **continuous action** (consisting of a vector of two entries) corresponds to the movement toward or away from the net and jumping.

The task is episodic, and in order to solve the environment, the agent must get an **average score of +0.5 over 100 consecutive episodes** (after taking the maximum over both agents). This means that after each episode we add up the rewards each agent received to get a score for each agent. At the end we will have two potential scores after an episode from which we take the maximum of those two. This score yields a single score for each episode.


Note: The project environment is similar to, but not identical to the Reacher environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

### Getting Started
The following instructions assume you are running on an Ubuntu 16.04 x64 distribution, with an [Anaconda](https://www.anaconda.com/download/#linux) installation. If you have installed Anaconda, please run following commands in order to execute the **Tennis** notebook:

Note: the python folder within this repository and the link to the pre-build Unity Environment for Linux are copied from the [Udacity DRLND course](https://github.com/udacity/deep-reinforcement-learning)
```bash
$ conda create --name drlnd python=3.6
$ source activate drlnd

$ git clone https://github.com/shelbyi/drlnd.git
$ cd drlnd/python
$ pip install .

$ python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
Next, download the pre-build Unity Environment for your Linux system.
<br>Linux visual: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
<br>Linux headless: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
<br>Place the file in the **project3-tennis** folder within the repository and unzip it.

Now, you should be set up to run the notebook on your system.

### Instructions
Before you run the notebook, be sure to change the kernel to **drlnd**. Execute cell after cell to get some information of the environment and to train both agenta to play tennis. At the end, the neural networks of the Actor & Critic model from both agents and the **replay buffer** which is shared by both agents are stored to disk under **[path to the repo]/drlnd/project3-tennis/models**.

Start the notebook and execute the cells.
```
$ cd [path to the repo]/drlnd/project3-tennis/
$ jupyter-notebook tennis.ipynb