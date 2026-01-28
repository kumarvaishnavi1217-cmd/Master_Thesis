import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:

    '''Manual Implementation of PPO Agent Memory

    ----------------
    Description
    ----------------

    This classes manages the memory of the manually implemented PPO algorithm

    ----------------
    Attributes:
    ----------------

    states: list
        A list of states collected during learning
        
    probs: list
        A list storing the probabilities of selecting actions during learning

    vals: list
        A list storing the values of states for learning purposes

    actions: list
        A list storing the actions chosen by the agent during learning

    rewards: list
        A list storing rewards collected by the agent during leaning

    dones: list
        A list storing flags indicating which states collected are terminal

    batch_size: int
        The number of inputs stored each batch chosen for learning will have

    ----------------
    Methods:
    ----------------

    generate_batches():
    store_memory(state, action, probs, vals, reward, done):
    clear_memory():
    '''

    def __init__(self, batch_size):
        '''PPO memory initialisation'''

        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        '''Generate training batches with samples for training'''

        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), \
                np.array(self.actions), \
                np.array(self.probs), \
                np.array(self.vals), \
                np.array(self.rewards), \
                np.array(self.dones), \
                batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        '''Takes in state, action, action probabilities, values of states, reward, terminal flag and stores them to memory'''

        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        '''Empties memory buffers'''
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class ActorNetwork(nn.Module):

    '''Implementation of Actor Network

    ----------------
    Description
    ----------------

    This classes builds the neural network of the actor for the PPO Agent

    ----------------
    Attributes:
    ----------------

    nF: int
        Number of neurons in the input layer of Actor Network
        It corresponds to the number of features in the environment observation
        
    nA: int
        Number of outputs of the Actor Network
        This number corresponds to the number of actions that agent can choose from

    linear_stack: _
        A sequential stacked model consisting of one input layer, two hidden layers, and one output layer
        Two kinds of activation functions are employed: ReLU and Softmax

    optimizer: _
        A list storing flags indicating which states collected are terminal

    device: _
        If cuda available device is 'cuda' else 'cpu'
        Cuda gives direct access to GPU

    ----------------
    Methods:
    ----------------

    forward(state: torch.Tensor) -> torch.Tensor:
    save_checkpoint(file_path):
    load_checkpoint(file_path):
    '''

    def __init__(self, num_features, num_actions, alpha ):
        '''Initialisation'''
        super(ActorNetwork, self).__init__()

        self.nF = num_features
        self.nA = num_actions

        self.linear_stack = nn.Sequential(
            nn.Linear(in_features = self.nF, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = self.nA),
            nn.Softmax(dim=1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        '''Takes state as input and returns probability distribution over the actions'''
        output = self.linear_stack(state)
        distribution = Categorical(output)
        return distribution
    
    def save_checkpoint(self, file_path):
        '''Save model parameters'''
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path):
        '''Load model parameters'''
        self.load_state_dict(torch.load(file_path))


class CriticNetwork(nn.Module):

    '''Implementation of Critic Network

    ----------------
    Description
    ----------------

    This classes builds the neural network of the critic for the PPO Agent

    ----------------
    Attributes:
    ----------------

    nF: int
        Number of neurons in the input layer of Actor Network
        It corresponds to the number of features in the environment observation
    
    linear_stack: _
        A sequential stacked model consisting of one input layer, two hidden layers, and one output layer
        Unlike actor model, critic only has one neuron in the output
        Two kinds of activation functions are employed: ReLU and Softmax

    optimizer: _
        A list storing flags indicating which states collected are terminal

    device: _
        If cuda available device is 'cuda' else 'cpu'
        Cuda gives direct access to GPU

    ----------------
    Methods:
    ----------------

    forward(state: torch.Tensor) -> torch.Tensor:
    save_checkpoint(file_path):
    load_checkpoint(file_path):
    '''

    def __init__(self, num_features, alpha):
        '''Initialisation'''
        super(CriticNetwork, self).__init__()

        self.nF = num_features

        self.linear_stack = nn.Sequential(
            nn.Linear(in_features = self.nF, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        '''Takes state as input and returns value function'''
        output = self.linear_stack(state)
        return output
    
    def save_checkpoint(self, file_path):
        '''Save model parameters to .pth file'''
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path):
        '''Load model parameters from .pth file'''
        self.load_state_dict(torch.load(file_path))


class PPOAgent():

    '''Manually implemented PPO Agent Class

    ----------------
    Description
    ----------------

    This classes builds the PPO Agent and all its relevant methods

    ----------------
    Attributes:
    ----------------

    gamma: float
        Discount factor
    
    policy_clip: float
        Clip parameter

    n_epochs: int
        Number of training epochs

    gae_lambda: float
        Generalized advantage estimate parameter

    entropy_coef: float
        Parameter that balances exploration and exploitation

    vf_coef: float

    actor: _
        Actor model

    critic: _
        Critic model

    memory: _

    ----------------
    Methods:
    ----------------

    remember(state, action, probs, vals, reward, done):
    save_models(path_actor, path_critic):
    load_models(path_actor, path_critic):
    choose_action(observation):
    evaluate(observation):
    learn():
    '''

    def __init__(self,
                 n_actions,
                 n_features,
                 gamma = 0.99,
                 alpha = 0.0003,
                 gae_lambda = 0.95,
                 entropy_coef = 0.0,
                 vf_coef = 0.5,
                 policy_clip = 0.2,
                 batch_size = 64,
                 n_epochs = 10
        ):
        '''Initialisation'''
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

        self.actor = ActorNetwork(num_features=n_features, num_actions=n_actions, alpha=alpha)
        self.critic = CriticNetwork(num_features=n_features, alpha=alpha)
        self.memory = PPOMemory(batch_size=batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        '''Save information collected to the PPO memory'''
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, path_actor, path_critic):
        '''Save actor and critic model parameters to files'''
        self.actor.save_checkpoint(path_actor)
        self.critic.save_checkpoint(path_critic)

    def load_models(self, path_actor, path_critic):
        '''Load actor and critic parameters from files'''
        self.actor.load_checkpoint(path_actor)
        self.critic.load_checkpoint(path_critic)

    def choose_action(self, observation):
        ''' Stochastic Policy: Select action depending on probability distribution from actor model'''
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    
    def evaluate(self, observation):
        '''Deterministic Policy: The action with highest probability is chosen'''
        state = torch.tensor(np.array([observation]), dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        action = dist.probs.argmax(dim=1).item()

        return action
    
    def learn(self):
        '''This method performs n_epochs learning iterations'''
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()
            
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage).to(self.actor.device)
            values = torch.tensor(values).to(self.actor.device) 

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                entropy = dist.entropy().sum()
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs) - self.entropy_coef * entropy
                actor_loss = actor_loss.mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + self.vf_coef * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        self.memory.clear_memory()













                

