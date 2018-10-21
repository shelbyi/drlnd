import numpy as np
import random
import copy

from .model import Actor, Critic
from .replaybuffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 lr_actor=0.001,
                 lr_critic=0.0001,
                 weight_decay_critic=0.00001,
                 replay_buffer_size=int(1e5),
                 batch_size=128,
                 update_every=5,
                 gamma=0.99,
                 tau=0.001,
                 seed=0,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 is_training=True,
                 load_prefix=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int):           state size of the task to solve
            action_size (int):          action size of the task to solve
            lr_actor (float):           learning rate for actor network
            lr_critic (float):          learning rate for critic network
            weight_decay_citic (float): weight decay rate for cirtic network
            replay_buffer_size (int):   replay buffer size
            batch_size (int):           minibatch size to train the neural network with
            gamma (int):                discount factor
            tau (int):                  for soft update of target parameters
            seed (int):                 define a seed to reproduce results
            device (string):            run on cpu or gpu (e.g. 'cuda:0' or 'cpu')
            is_training (boolean):      parameter that defines if it is training or playing mode
            load_prefix (string):       prefix-name of networks that will be loaded
        """

        print(device)
        self._state_size = state_size
        self._action_size = action_size
        self._lr_actor = lr_actor
        self._lr_critic = lr_critic
        self._weight_decay_critic = weight_decay_critic
        self._replay_buffer_size = replay_buffer_size
        self._batch_size = batch_size
        self._update_every = update_every
        self._gamma = gamma
        self._tau = tau
        self._seed = random.seed(0)
        self._device = device

        # Actor Network (w/ Target Network)
        self._actor_local = Actor(self._state_size, self._action_size, seed).to(self._device)
        self._actor_target = Actor(self._state_size, self._action_size, seed).to(self._device)
        self._actor_optimizer = optim.Adam(self._actor_local.parameters(), lr=self._lr_actor)

        # Critic Network (w/ Target Network)
        self._critic_local = Critic(self._state_size, self._action_size, seed).to(self._device)
        self._critic_target = Critic(self._state_size, self._action_size, seed).to(self._device)
        self._critic_optimizer = optim.Adam(self._critic_local.parameters(), lr=self._lr_critic, weight_decay=self._weight_decay_critic)

        self._hard_copy(self._actor_target, self._actor_local)
        self._hard_copy(self._critic_target, self._critic_local)

        # Noise process
        self._noise = OUNoise(self._action_size, seed)

        if not is_training:
            self._load(load_prefix)

        # Replay memory
        self._memory = ReplayBuffer(self._action_size,
                                    self._replay_buffer_size,
                                    self._batch_size,
                                    seed,
                                    self._device,
                                    False)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self._t_step = 0

    @classmethod
    def for_training(cls,
                     state_size,
                     action_size,
                     lr_actor=0.001,
                     lr_critic=0.0001,
                     weight_decay_critic=0.00001,
                     replay_buffer_size=int(1e5),
                     batch_size=128,
                     update_every=5,
                     gamma=0.99,
                     tau=1e-3,
                     seed=0,
                     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        return cls(state_size, action_size, lr_actor, lr_critic, weight_decay_critic,
                   replay_buffer_size, batch_size, update_every, gamma, tau, seed, device,
                   is_training=True)

    @classmethod
    def for_playing(cls, state_size, action_size, prefix, seed=0):
        return cls(state_size, action_size,
                   is_training=False, load_prefix=prefix, seed=seed)

    def _hard_copy(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self._memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self._t_step = (self._t_step + 1) % self._update_every
        if self._t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self._memory) > self._batch_size:
                experiences = self._memory.sample()
                self._learn(experiences, self._gamma)

    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._actor_local.eval()
        with torch.no_grad():
            action = self._actor_local(state).cpu().data.numpy()
        self._actor_local.train()
        if add_noise:
            action += self._noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self._noise.reset()

    def _learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, probabilities, idxs = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self._actor_target(next_states)
        Q_targets_next = self._critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self._critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self._actor_local(states)
        actor_loss = -self._critic_local(states, actions_pred).mean()
        # Minimize the loss
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self._soft_update(self._critic_local, self._critic_target, self._tau)
        self._soft_update(self._actor_local, self._actor_target, self._tau)

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, prefix):
        self._save_model(prefix)
        self._save_replay_buffer(prefix)

    def _save_model(self, prefix):
        torch.save(self._actor_local.state_dict(), prefix + 'actor_local.pth')
        torch.save(self._actor_target.state_dict(), prefix + 'actor_target.pth')

        torch.save(self._critic_local.state_dict(), prefix + 'critic_local.pth')
        torch.save(self._critic_target.state_dict(), prefix + 'critic_target.pth')

    def _save_replay_buffer(self, prefix):
        self._memory.save(prefix)

    def _load(self, prefix):
        self._actor_local.load_state_dict(torch.load(prefix + 'actor_local.pth'))
        self._actor_target.load_state_dict(torch.load(prefix + 'actor_target.pth'))

        self._critic_local.load_state_dict(torch.load(prefix + 'critic_local.pth'))
        self._critic_target.load_state_dict(torch.load(prefix + 'critic_target.pth'))


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self._mu = mu * np.ones(size)
        self._theta = theta
        self._sigma = sigma
        self._seed = random.seed(seed)
        self.reset()
        self._state = None

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self._state = copy.copy(self._mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self._state
        dx = self._theta * (self._mu - x) + self._sigma * np.array([random.random() for i in range(len(x))])
        self._state = x + dx
        return self._state
