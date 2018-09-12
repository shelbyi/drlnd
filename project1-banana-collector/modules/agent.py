import random
import numpy as np

import torch

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from .replaybuffer import ReplayBuffer
from .qnetwork import QNetwork


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 activate_ddqn=False,
                 activate_prioritized_experience_replay=False,
                 replay_buffer_size=int(1e5),
                 batch_size=64,
                 learning_rate=5e-4,
                 update_every=13,
                 gamma=0.99,
                 tau=1e-3,
                 seed=0,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 is_training=True,
                 load_prefix=None):
        
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int):         state size of the task to solve
            action_size (int):        action size of the task to solve
            activateDDQN (boolean):   boolean to activate/deactivate double DQN
            activatePrioritizedExperienceReplay (boolean): boolean to activate/deactivate prioritized experience replay
            replay_buffer_size (int): replay buffer size
            batch_size (int):         minibatch size to train the neural network with
            learning_rate (int):      learning rate
            update_every (int):       how often to update the network
            gamma (int):              discount factor
            tau (int):                for soft update of target parameters
            seed (int):               define a seed to reproduce results
            device (string):          run on cpu or gpu (e.g. 'cuda:0' or 'cpu')
        """
        print(device)
        self._state_size = state_size
        self._action_size = action_size
        self._activateDDQN = activate_ddqn
        self._activatePrioritizedExperienceReplay = activate_prioritized_experience_replay
        self._replay_buffer_size = replay_buffer_size
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._update_every = update_every
        self._gamma = gamma
        self._tau = tau
        self._seed = random.seed(seed)
        self._device = device

        self.PER_b = 0.001

        # Q-Network
        self._qnetwork_local = QNetwork(self._state_size, self._action_size, seed).to(self._device)
        self._qnetwork_target = QNetwork(self._state_size, self._action_size, seed).to(self._device)  # target network ist "fest"
        self._optimizer = optim.Adam(self._qnetwork_local.parameters(), lr=self._learning_rate)

        if not is_training:
            self._load(load_prefix)

            # Replay memory
        self._memory = ReplayBuffer(self._action_size, 
                                    self._replay_buffer_size, 
                                    self._batch_size, 
                                    seed, 
                                    self._device,
                                    self._activatePrioritizedExperienceReplay)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self._t_step = 0

    @classmethod
    def for_training(cls,
                     state_size,
                     action_size,
                     activate_ddqn=False,
                     activate_prioritized_experience_replay=False,
                     replay_buffer_size=int(1e5),
                     batch_size=64,
                     learning_rate=5e-4,
                     update_every=13,
                     gamma=0.99,
                     tau=1e-3,
                     seed=0,
                     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        return cls(state_size, action_size, activate_ddqn, activate_prioritized_experience_replay,
                   replay_buffer_size, batch_size, learning_rate, update_every, gamma, tau, seed, device,
                   is_training=True)

    @classmethod
    def for_playing(cls, state_size, action_size, prefix, seed=0):
        return cls(state_size, action_size,
                   is_training=False, load_prefix=prefix, seed=seed)
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self._memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self._t_step = (self._t_step + 1) % self._update_every
        if self._t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self._memory) > self._batch_size:
                experiences = self._memory.sample()
                self._learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======cself._self._
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._qnetwork_local.eval()
        with torch.no_grad():
            action_values = self._qnetwork_local(state)
        self._qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self._action_size))

    def _learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, probabilities, idxs = experiences

        if self._activateDDQN:
            _, next_state_actions = self._qnetwork_local(next_states).detach().max(1, keepdim=True)
            Q_targets_next = self._qnetwork_target(next_states).gather(1, next_state_actions)
        else:
           # Get max predicted Q values (for next states) from target model
            Q_targets_next = self._qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self._gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self._qnetwork_local(states).gather(1, actions)
        
        if self._activatePrioritizedExperienceReplay:
            absolute_errors = torch.abs(Q_targets - Q_expected)
            self._memory.update(idxs, (Variable(absolute_errors).data).cpu().numpy()[0])

            # TODO: update self.PER_b over learning time 
            importantSamplingWeights = torch.pow(self._replay_buffer_size * probabilities, -self.PER_b)
            importantSamplingWeights /= importantSamplingWeights.max()

            loss = torch.sum((importantSamplingWeights * (Q_targets - Q_expected) ** 2))
        else:
            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # ------------------- update target network ------------------- #
        self._soft_update(self._qnetwork_local, self._qnetwork_target)                     

    def _soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self._tau*local_param.data + (1.0-self._tau)*target_param.data)
            
    def save(self, prefix):
        self._save_model(prefix)
        self._save_replay_buffer(prefix)
    
    def _save_model(self, prefix):
        torch.save(self._qnetwork_local.state_dict(), prefix + 'qnetwork_local.pth')
        torch.save(self._qnetwork_target.state_dict(), prefix + 'qnetwork_target.pth')

    def _save_replay_buffer(self, prefix):
        self._memory.save(prefix)
        
    def _load(self, prefix):
        self._qnetwork_local.load_state_dict(torch.load(prefix + 'qnetwork_local.pth'))
        self._qnetwork_target.load_state_dict(torch.load(prefix + 'qnetwork_target.pth'))