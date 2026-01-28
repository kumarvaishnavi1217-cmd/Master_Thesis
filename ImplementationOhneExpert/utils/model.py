import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.categorical import Categorical
from typing import Callable, Tuple
from stable_baselines3.common.policies import ActorCriticPolicy
from gym import spaces


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Custom Acctor and Critic Policy Networks for PPO
'''

class CustomNetwork(nn.Module):

    '''
    Critic and Actor Model Constructor
    
    This class builds the Critic and the Actor Networks
    The two models differ in the number of neurons in the output layer and
    The actor has a Softmax activation function in the output layer to
    give a probability for each action that may be selected.
    
    '''

    def __init__(
        self,
        feature_dim: int,
        nA: int,
        nV: int
    ):
        '''Initialisation'''

        super().__init__()

        self.latent_dim_pi = nA
        self.latent_dim_vf = nV

        # Actor (Policy) network
        self.policy_net = nn.Sequential(
            nn.Linear(in_features = feature_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = nA),
            nn.Softmax(dim=1)
        )

        # Critic (Value) network
        self.value_net = nn.Sequential(
            nn.Linear(in_features = feature_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = nV)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Forward pass of both Actor and Critic'''
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        '''Forward pass Actor'''
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        '''Forward pass of Critic'''
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):

    ''' Custom Policy Builder necessary for creating a custom policy to use when implementing PPO with Stable Baselines 3'''

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, nA=11, nV=11) # nA = nV


'''
Recurrent Neural Network for Sequential Mode - (this was not extensively tested)
'''

class LSTMNet(nn.Module):

    '''
    Long-Short-Term Memory Network that learns patterns in sequential data

    In this project, this was not extensively experimented with. The Idea behind its
    use, was to find relationship behind driving behaviour and the characteristics
    of the track ahead of the current train position.
    '''

    def __init__(self, 
            input_size, 
            hidden_size,
            output_size
        ):
        '''Iniitialisation'''
        
        super(LSTMNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        '''Forward pass, takes in sequence and returns an output array'''
        
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm1(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.bn(out)
        out = nn.functional.relu(out)

        return out


class ATDNETv14(nn.Module):

    '''
    Combination of LSTM Nets and Sequential Linear Stack 

    This model was not extensively experimented with.
    Idea: Two seperate LSTM Networks. One learns patterns necessary for train operation in the speed limits of the track ahead.
    the other learns patterns in the slope profile of the track ahead of the current train position.
    '''

    def __init__(self,
            num_nonseq_features = 5,
            output_size = 4,
            num_actions = 11
        ):
        '''initialisation'''

        super(ATDNETv14, self).__init__()

        self.nnsF = num_nonseq_features
        self.nOS = output_size
        self.nA = num_actions
        self.linstack_input_size = (2 * self.nOS) + self.nnsF

        # LSTMNet for the future speed limit segments on the track until destination
        self.LSTMNet_limits = LSTMNet(
            input_size = 2,
            hidden_size = 8,
            output_size = self.nOS
        )

        # LSTMNet for the future gradient segement on the track until destination
        self.LSTMNet_gradients = LSTMNet(
            input_size = 2,
            hidden_size = 8,
            output_size = self.nOS
        )

        # linear stack of linear layers with ReLU and Softmax activation layers
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features = self.linstack_input_size, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features = 64, out_features = 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features = 32, out_features = self.nA),
            nn.Softmax(dim=1)
        )

    def forward(self, x_state_batch) -> torch.Tensor:
        '''Forward Pass'''

        x_state_1_batch = []
        x_state_speed_limits_batch = []
        x_state_gradients_batch = []

        inputs = []

        for x_state in x_state_batch:

            x_state_1_batch.append(x_state[0])
            x_state_speed_limits_batch.append(x_state[1])
            x_state_gradients_batch.append(x_state[2])

        tensor_1 = torch.stack(x_state_1_batch)
        padded_sequences_limits = pad_sequence(x_state_speed_limits_batch, batch_first=True)
        padded_sequences_slopes = pad_sequence(x_state_gradients_batch, batch_first=True)

        inputs.append(tensor_1)
        inputs.append(self.LSTMNet_limits(padded_sequences_limits))
        inputs.append(self.LSTMNet_gradients(padded_sequences_slopes))

        combined_input = torch.cat(inputs, dim=1)

        out = self.linear_stack(combined_input)

        return out
    

'''
Simple MLP Model for DDQN - (used for non sequential training mode.)
'''

class ATDNETv18(nn.Module):

    '''
    Simple Sequential linear stack

    This model is used for the DDQN-Learning algorithm.
    Designed for non sequential data. Same state feature space as that applied for PPO-Learning algorithm
    '''
    
    def __init__(self,
            num_features,
            num_actions
        ):
        '''Initialisation'''

        super(ATDNETv18, self).__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(in_features = num_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = 256),
            nn.ReLU(),
            nn.Linear(in_features = 256, out_features = num_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        '''Foraward Pass'''

        out = self.linear_stack(state)
        return out