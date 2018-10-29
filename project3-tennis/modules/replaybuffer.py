import random
import pickle
import numpy as np

import torch

from collections import deque


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, 
                 action_size, 
                 buffer_size, 
                 batch_size, 
                 seed, 
                 device,
                 activate_prioritized_experience_replay):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (string):          run on cpu or gpu (e.g. 'cuda:0' or 'cpu')
            activatePrioritizedExperienceReplay (boolean): true, if prioritized experience activate
        """
        self._action_size = action_size
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._seed = random.seed(seed)
        self._device = device
        self.activate_prioritized_experience_replay = activate_prioritized_experience_replay
        
        self._memory = deque(maxlen=buffer_size)
        self._epsilon = 0.001
        
        self.idx = -1
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.idx = (self.idx + 1) % self._buffer_size
        
        priority = 1.0 + self._epsilon
        e = [state, action, reward, next_state, done, priority, self.idx]
        self._memory.append(e)
        
    def update(self, idxs, priorities):
        if self.activate_prioritized_experience_replay:
            for idx, priority in zip(idxs, priorities):
                for experience in self._memory:
                    #print(idx)
                    #print(experience[2])
                    if idx == experience[6]:
                        # print(experience)
                        experience[5] = priority
                        # print(experience)
                        # print("###################################")
                        break
        else:
            pass
                        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        my_dist = []
        my_list = []

        mySum = np.sum([num[5] for num in self._memory])
            
        for row in self._memory:
            my_list.append(row)
            my_dist.append(row[5] / mySum)
                
        if self.activate_prioritized_experience_replay:
            experiences_indices = np.random.choice(len(self._memory), size=self._batch_size, p=my_dist).tolist()
            experiences = np.array(my_list)[experiences_indices]
            
            experiences_dist = np.array(my_dist)[experiences_indices]
            probabilities = torch.from_numpy(np.vstack([e for e in experiences_dist if e is not None])).float().to(self._device)

        else:
            experiences = random.sample(my_list, k=self._batch_size)
            probabilities = torch.from_numpy(np.vstack([e[5] for e in experiences if e is not None])).float().to(self._device)
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self._device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(self._device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self._device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self._device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self._device)

        idxs = torch.from_numpy(np.vstack([e[6] for e in experiences if e is not None])).long().to(self._device)

        return (states, actions, rewards, next_states, dones, probabilities, idxs)

    def save(self, prefix):
        with open(prefix + 'replaybuffer.pkl', 'wb') as output:
            pickle.dump(self._memory, output, pickle.HIGHEST_PROTOCOL)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self._memory)