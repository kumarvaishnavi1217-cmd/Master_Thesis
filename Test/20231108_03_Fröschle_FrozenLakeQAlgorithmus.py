import gymnasium as gym
import numpy as np 

# 1. Load Environment and Q-table structure
env = gym.make('FrozenLake8x8-v1')

Q = np.zeros([env.observation_space.n,env.action_space.n])
# env.observation.n, env.action_space.n gives number of states and action in env loaded

# 2. Parameters of Q-learning
alpha = 0.8     
gamma = 0.95   
epsilon = 0.1  
episodes = 2000
rewards_list = [] # rewards per episode calculate

# 3. Q-learning Algorithm
for i in range(episodes):
    # Reset environment
    state, info = env.reset() 
    rewardsEpisode = 0

    terminated = False
    truncated = False
    
    #The Q-Table learning algorithm
    while not terminated and not truncated:                                 
        env.render() 

        # Choose action from Q table
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Zufällige Aktion wählen
        else:
            action = np.argmax(Q[state])  # Beste Aktion wählen                       
        
        #Get new state & reward from environment
        observation,reward,terminated,truncated,info = env.step(action)

        #Update Q-Table with new knowledge
        Q[state,action] = Q[state,action] + alpha*(reward + gamma*np.max(Q[observation,:]) - Q[state,action])
        rewardsEpisode += reward
        state = observation

    rewards_list.append(rewardsEpisode)
    env.render() 
    
print("Reward Sum on all episodes " + str(sum(rewards_list)/episodes))
print("Final Values Q-Table")
print(Q)