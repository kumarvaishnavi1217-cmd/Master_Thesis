import random
from collections import namedtuple, deque
import torch
import numpy as np


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    '''Replay Memory for Double Deep Q Learning

    ----------------
    Attributes:
    ----------------

    memory: deque
        Deque with fixed maximum size working as a circular memory

    batch_size: int
        Number of transitions per batch

    sequence_length: int
        Number of consecutive transistions per sequence

    default_transition: namedtuple
        A default transistion to fill gaps wherever necessary

    device: _
        'cuda' or 'cpu' depending on availability of cuda for accessing GPU

    ----------------
    Methods:
    ----------------

    push(*args)
    sampling_possible(type='nonsequential')
    sample(recurrent_agent = False)
    '''

    def __init__(self, capacity, batch_size=0, sequence_length=0, num_features=0, device='cpu'):
        '''Initialisation'''

        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.default_transition = Transition(np.zeros(num_features), 0, np.zeros(num_features), 0, False)
        self.device = device


    def push(self, *args):
        '''Save a transition to memory'''
        self.memory.append(Transition(*args))


    def sampling_possible(self, type='nonsequential'):
        '''This method checks if there are enough transistions available in the memory to perform a sampling'''

        assert type in ['sequential', 'nonsequential'], \
            "type should be either 'sequential' or 'nonsequential'."

        if type == 'nonsequential':
            return len(self.memory) > self.batch_size

        elif type == 'sequential':
            return len(self.memory) > self.batch_size + self.sequence_length 
            

    def sample(self, recurrent_agent = False):
        '''This method samples a batch of random transitions for training.
        
        recurrent_agent = True:
            Sample data for training the recurrent agent
        
        recurrent_agent = False:
            Sample data for training the non recurrent agent
        '''
        
        assert not (self.batch_size == 0), \
            "batch_size cannot be zero (0). use method set_batch_size() \
                to set batch size."

        sampled_transitions = random.sample(self.memory, self.batch_size)

        states = []
        next_states = []
        rewards = []
        actions = []
        dones = []

        for transition in sampled_transitions:

            states.append(transition.state)
            actions.append(transition.action)
            next_states.append(transition.next_state)
            rewards.append(transition.reward)
            dones.append(transition.done)

        if not recurrent_agent:

            return torch.FloatTensor(np.array(states)).to(self.device), torch.FloatTensor(np.array(next_states)).to(self.device), \
                    torch.IntTensor(np.array(actions)).to(self.device), torch.FloatTensor(np.array(rewards)).to(self.device), \
                    torch.IntTensor(np.array(dones)).to(self.device)
        
        else:

            new_states = []
            new_next_states = []

            for state in states:
                new_states.append([torch.FloatTensor(state[0]).to(self.device), \
                                   torch.FloatTensor(np.array(state[1])).to(self.device), \
                                   torch.FloatTensor(np.array(state[2])).to(self.device)])
                
            for next_state in next_states:
                new_next_states.append([torch.FloatTensor(next_state[0]).to(self.device), \
                                   torch.FloatTensor(np.array(next_state[1])).to(self.device), \
                                   torch.FloatTensor(np.array(next_state[2])).to(self.device)])
                
            return new_states, new_next_states, \
                    torch.IntTensor(np.array(actions)).to(self.device), torch.FloatTensor(np.array(rewards)).to(self.device), \
                    torch.IntTensor(np.array(dones)).to(self.device)
        
