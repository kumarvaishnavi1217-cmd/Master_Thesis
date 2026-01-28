import torch
import numpy as np
import random
from gym_train.gym_train.envs import TrainEnv
from torchsummary import summary

from utils.replaymemory import *
from utils.model import ATDNETv14, ATDNETv18


class DDQNAgent():

    '''
    Class for DQN or Double-DQN model Architecture.

    This class constructs the agent for DDQN. There is an option to also use it as a DQN agent too.

    Attributes
    ----------
    device: str - 'cuda' if GPU is available else 'cpu'
    env: TrainEnv - Environment with which the agent will interact
    discount_factor: float
    batch_size: int
    learning_frquency: int - How often the Q model will be updated
    target_update_frequency: int - How often the Target Q network parameters will be copied from the Q model
    learning_starts: int - Number of steps after which the agent will start learning
    memory: ReplayMemory
    Q: Model of agent's Q network
    Q_target: Model of Target Q network
    optimizer: _ - Optimization technique used in updating model parameters
    loss_function: _ - Function used to compute the loss between current Q values and Target Q Values
    init_epilon: int - The initial exploration - exploitation balance parameter
    final_epsilon: int - The final exploration - exploitation balance parameter
    max_episodes: int - Number of Episodes to train agent for
    max_episode_length: int - Maximum number of steps per episode
    double_dqn: bool - True if DDQN, False if DQN

    '''

    def __init__(self, env:TrainEnv, gamma, batch_size, learn_freq, tar_update_freq, mem_capacity, \
                 max_len, learning_starts, num_episodes=1e6):
        '''
        agent = DDQNAgent(env=env, gamma=discount_factor, batch_size=batch_size, learn_freq=learning_frequency, \
                              tar_update_freq=tar_update_freq, mem_capacity=mem_capacity, learning_starts=learning_starts,
                              num_episodes=num_episodes, max_len=max_episode_length)
        '''

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.env = env
        self.discount_factor = gamma
        self.batch_size = batch_size
        self.learning_frequency = learn_freq
        self.target_update_frequency = tar_update_freq
        self.learning_starts = learning_starts
        self.memory = ReplayMemory(capacity = mem_capacity, batch_size=self.batch_size, \
            num_features=env.nF, device=self.device)
        self.Q = ATDNETv18(env.nF, env.nA).to(self.device) \
            if not self.env.sequential_state else ATDNETv14(num_actions=env.nA).to(self.device)
        # summary(self.Q, input_size=[(5,), (12, 2), (10, 2)])
        self.Q_target = ATDNETv18(env.nF, env.nA).to(self.device) \
            if not self.env.sequential_state else ATDNETv14(num_actions=env.nA).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q.parameters())
        self.loss_function = torch.nn.MSELoss()
        self.init_epsilon = 1.0
        self.final_epsilon = 0.1
        self.max_episodes = num_episodes
        self.max_episode_length = max_len
        self.episode_start_decay = 0
        self.double_dqn = True


    def get_action(self, state, epsilon, with_expert):

        '''
        This method will return an action selected either by the agent model or a random action
        depending on the exploration parameter, epsilon
        '''

        # Get a random value. If it's greater than epsilon, action chosen by agent
        if random.random() > epsilon:
            if not self.env.sequential_state:
                self.Q.eval()
                state_tensor = torch.unsqueeze(torch.FloatTensor(np.array(state)), dim=0).to(self.device)
                return torch.argmax(self.Q(state_tensor).data, dim=-1).data.item()
            else:
                self.Q.eval()
                state_1 = torch.Tensor(state[0]).to(self.device)
                state_limits = torch.Tensor(np.array(state[1])).to(self.device)
                state_slopes = torch.Tensor(np.array(state[2])).to(self.device)
                state_tensor = [[state_1, state_limits, state_slopes]]
                return torch.argmax(self.Q(state_tensor).data, dim=-1).data.item()
            
        # If the random value isn't greater than epsilon, action is chosen randomly
        else:
            return self.env.action_space.sample()


    def get_epsilon(self, current_step_counter, current_episode):

        '''
        This method implements the exploration-exploitation balance by choosing the epsilon parameter
        depending on the steps and episodes of learning done

        The agent has two learning phases:

        Phase 1:
            As long as there is not enough training samples in the Replay Buffer, just explore
        Phase 2:
            If there is enough data to start training, start dropping epsilon value linearly per new episode
        '''
        
        # Phase 1 - Only Exploration
        if current_step_counter < self.learning_starts: 
            self.episode_start_decay = current_episode
            return self.init_epsilon
        # Phase 2 - Exploration decay
        else: 
            return self.init_epsilon - (0.9/(self.max_episodes - self.episode_start_decay))*(current_episode - self.episode_start_decay)


    def update_Q(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):

        '''
        This method computes the MSE loss between the current Q value predictions and target Q values (Ground Truth).
        Two learning algorithms are supported: 'DQN' and 'DDQN'.
        '''

        actions = action_batch.type(torch.int64)
        not_done_mask = torch.sub(1, terminal_batch)

        '''
        Current Q(s,a) Values -- Prediction
        '''
        current_Q_values = self.Q(state_batch).gather(1, actions.unsqueeze(1)).squeeze()

        if self.double_dqn:
            '''
            Double-DQN Target -- Ground Truth
            '''
            # target = r' + gamma * Q_target(s', argmax_a Q(s',a))
            next_Q_values = self.Q(next_state_batch)
            next_Q_target_values = self.Q_target(next_state_batch)
            next_Q_values_max = not_done_mask * next_Q_target_values.gather(1, torch.argmax(next_Q_values, dim=-1).unsqueeze(1)).detach().squeeze()

            target_Q_values = reward_batch + (self.discount_factor * next_Q_values_max)
            
        else:
            '''
            DQN Target -- Ground Truth
            '''
            # target = r' + gamma * max_a Q_target(s', a)
            next_max_q_target = self.Q_target(next_state_batch).detach().max(1)[0]
            next_Q_target_values = not_done_mask * next_max_q_target
            
            target_Q_values = reward_batch + (self.discount_factor * next_Q_target_values)

        loss = self.loss_function(current_Q_values, target_Q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_Q_target(self):

        '''
        copy parameter of Q network to Q_target network
        '''
        
        self.Q_target.load_state_dict(self.Q.state_dict())


    def learn(self, step, replay_rounds):

        '''
        This methods does on iteration of learning - One episode
        '''

        # Learning Phase
        if self.memory.sampling_possible(type='nonsequential') and \
            step > self.learning_starts and \
            step % self.learning_frequency == 0:

            x, x_next, actions, rewards, terminals = self.memory.sample(recurrent_agent=self.env.sequential_state)
            replay_rounds += 1

            self.update_Q(state_batch=x, action_batch=actions, \
                reward_batch=rewards, next_state_batch=x_next, terminal_batch=terminals)

        # Target Update
        if step > self.learning_starts and \
            step % self.target_update_frequency == 0:

            self.update_Q_target()

