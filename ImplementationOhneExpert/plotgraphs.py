'''
This .py script just helped plot graphs during experimentation
'''

# import matplotlib.pyplot as plt
# import numpy as np


'''
plot Resistance against Speed
'''

# speeds = []
# resistances = []
# resistances_2 = []
# f_airs = []
# tractions = []
# As = []
# Bs = []
# Cs = []
# As_2 = []
# Bs_2 = []
# Cs_2 = []

# action = 100

# totalpower = 2000
# S = 4.4 * 3.2
# d = 2 * (3.9 + 3.2)
# l = 75.202
# l_g = 0.51
# C_x = 0.45
# mass = 140
# m_TC = 0.5 * mass
# m_PC = 0.5 * mass

# totalpower = 2600
# n_PC = 2
# n_TC = 2
# n_p = 2
# n_B = 5
# S = 3.9 * 3.2
# d = 2 * (3.9 + 3.2)
# l = 75.2
# l_g = 0.51
# C_x = 0.7
# mass = 140
# m_TC = 0.5 * mass
# m_PC = 0.5 * mass

# c1 = 0.45
# c2 = 0.45
# c3 = 0.455/(np.log10(19000*l*80))
# c4 = 0.0004
# c5 = 0.0002
# c6 = 0.002
# c7 = 0.0005

# # plt.figure(figsize=(10,3))

# for speed in np.arange(0, 171, .1):

#     speeds.append(speed)
#     nspeed = speed / 3.6
    
#     A = 6.4*m_TC + 8.0*m_PC
#     Bv = (0.18*mass + n_TC + 0.005*n_PC*totalpower) * nspeed
#     Cv2 = (0.6125*C_x*S + 0.00197*d*l + 0.0021*d*l_g*(n_PC + n_TC - 1) + 0.2061*C_x*n_B + 0.2566*n_p) * (nspeed ** 2)

#     A_2 = 0.008232 * mass
#     B_2 = 0 * nspeed
#     C_2 = (2.3324 + 5.5057 * 4) * (nspeed ** 2)

#     F_air = ((c1 + c2) * S + (c3 + c4 + c5 + c6 + c7) * (l * d) - (3*l_g*d))* (1.25/2) * nspeed ** 2

#     traction = (action/100) * min(200000, 2000000/(nspeed+0.001))
#     coefficient = 0.161 + (7.5/(speed + 44))
#     tractions.append(min(m_PC*1000*coefficient*9.81, traction)/1000)


#     As.append(A/1000)
#     Bs.append(Bv/1000)
#     Cs.append(Cv2/1000)

#     As_2.append(A_2/1000)
#     Bs_2.append(B_2/1000)
#     Cs_2.append(C_2/1000)

#     resistances.append((A + Bv + Cv2)/1000)
#     resistances_2.append((A_2 + B_2 + C_2)/1000)
#     f_airs.append(F_air/1000)


# # radii = []
# # f_curs_1 = []
# # f_curs_2 = []
# # mu = 0.21
# # s_k = 1.524  # Stützweite (Gauge)
# # a_s = 2.75  # Achsabstand (Distance between axles)

# # for radius in np.arange(201.1, 10001.0, 200.0):

# #     radii.append(radius)
# #     f_curs_1.append(132000 * 9.81 * mu * ((0.72*s_k + 0.47*a_s)/radius))
# #     if radius >= 300:
# #         f_curs_2.append(132 * 9.81 * (500/(radius - 30)))
# #     else:
# #         f_curs_2.append(132 * 9.81 * (650/(radius - 55)))


# # plt.plot(speeds, As, label='A')
# # plt.plot(speeds, Bs, label='Bv')
# # plt.plot(speeds, Cs, label='Cv^2')
# # plt.plot(speeds, resistances, label='F_res')

# # plt.plot(speeds, As_2, label='A_2', linestyle='dashed')
# # plt.plot(speeds, Bs_2, label='Bv_2', linestyle='dashed')
# # plt.plot(speeds, Cs_2, label='Cv^2_2', linestyle='dashed')
# plt.plot(speeds, resistances_2, label='Motion Resistance')
# # plt.plot(speeds, tractions, label='Traction Effort')
# plt.grid()

# plt.xlabel('Speed [km/hr]')
# plt.ylabel('F_res [kN]')
# plt.title('Motion Resritance')

# # plt.plot(radii, f_curs_1, label='Protopapadakis')
# # plt.plot(radii, f_curs_2, label='Roeckl', linestyle='dashed')
# plt.show()


'''
Energy Reward
'''

# r_energy_s = []
# speeds = []

# mass = 132000
# rmc = 1.2
# F_max = 132000
# a_hold_time = 2
# max_speed = 44.44
# eff_Mass = mass * rmc
# max_acceleration = F_max / eff_Mass

# while F_max > 0:

#     for speed in np.arange(0.0, 44.8, 0.1):
        
#         E_max = max_speed * max_acceleration * a_hold_time
#         E = (F_max/eff_Mass) * abs(speed) * a_hold_time
        
#         speeds.append(speed)
#         r_energy_s.append(1 - 0.2 * (E/E_max))

#     plt.plot(speeds, r_energy_s, label='F = '+str(F_max/1000)+' kN')

#     F_max -= 2e4
#     speeds = []
#     r_energy_s = []

# plt.xlabel('Speed [m/s]')
# plt.ylabel('Energy Reward')

# plt.legend()
# plt.show()

'''
Speed Following Reward
'''

# import math 

# r_speed_s = []
# speeds = []

# for delta_speed in np.arange(-20.0, 20.1, 0.1):
#     r_speed = 1.0 - 1.0 / (1.0 + math.exp(-.8 * (abs(delta_speed) - 5.0)))
    
#     speeds.append(delta_speed)
#     r_speed_s.append(r_speed)

# plt.plot(speeds, r_speed_s)

# plt.xlabel('Speed - Estimated Speed [m/s]')
# plt.ylabel('Speed Following Reward')

# plt.show()

'''
Punctuality reward dependent on expended Energy
'''

# import math 

# r_time_energy_s = []
# delta_times = []

# for E in np.arange(0.0, 0.1, 0.2):

#     for delta_time in np.arange(0.0, 300.1, 0.1):

#         r_energy = 1 # - E
#         r_time_energy = max(-1.0, (1 - (delta_time/60)**0.63) * r_energy)
        
#         delta_times.append(delta_time)
#         r_time_energy_s.append(r_time_energy)

#     plt.plot(delta_times, r_time_energy_s, label='E = ' + str(round(E,1)))

#     delta_times = []
#     r_time_energy_s = []

# plt.xlabel('punctuality error [s]')
# plt.ylabel('punctuality reward')

# plt.grid()
# plt.show()


'''
Parking
'''

# def f(x, y):
#     return np.exp(-1/(x*y))

# # Create a grid of x and y values
# x = np.linspace(0.001, 1, 2000)
# y = np.linspace(0.001, 1, 44.4)
# X, Y = np.meshgrid(x, y)

# # Evaluate the function for all x and y values
# Z = np.exp(-1/(X*Y))

# # Create a 3D plot

# plt.contour(X, Y, Z, cmap='plasma')

# # Add labels and title
# plt.set_xlabel('Distance to Destination')
# plt.set_ylabel('Speed ')
# plt.set_zlabel('f(x,y)')
# plt.set_title('Parking Reward')

# # Show the plot
# plt.show()

# x = np.linspace(0, 100000, 100000)
# y = [-1 * ((i)/94000) ** 0.4 if i > 5 else 1.0 for i in x]  # if i > 5 else 1.0 

# plt.plot(x, y)
# plt.xlabel('distance to destination [m]')
# plt.ylabel('reward')
# plt.title('Parking Reward')
# plt.show()

'''
Punctuality dependent on speed at destination
'''

# two_D = False

# if two_D:
#     x = np.linspace(0, 600, 1000)
#     for speed in np.arange(0.0, 0.1, 10.0):
#         y_p = [max(1-((i/600)**2 + (speed/50)**2) ** 0.2, 0) for i in x]

#         plt.plot(x, y_p, label = str(speed) + ' m/s')

#     plt.xlabel('punctuality error [s]')
#     plt.ylabel('reward')
#     plt.title('Punctuality Reward')
#     plt.legend()
#     plt.show()

# else:

#     x = np.linspace(0.0, 180.0, 100)
#     y = np.linspace(0.0, 15.0, 100)

#     X, Y = np.meshgrid(x, y)

#     # Z = (1 - (X/60)**0.63) * (1 - (Y/5)**0.63)
#     Z = 1 - ((X/60)**2 + (Y/5)**2)**0.63

#     plt.contourf(X, Y, Z, 50, cmap='RdGy')
#     plt.colorbar()

#     plt.xlabel('Punctuality Error [s]')
#     plt.ylabel('Parking Error [m]')
#     plt.title('Terminal Reward')

#     plt.show()


'''
Plot Learning Progress
'''

# import csv
# import numpy as np
# import matplotlib.pyplot as plt

# # Read the CSV file
# with open('models/TestingPPO_20230412_01_NoRNN/logging.csv', 'r') as file:
#     reader = csv.reader(file)
#     data = list(reader)

# # Extract the x and y data from the CSV file
# x_data = [float(row[0]) for row in data]
# y_data = [float(row[1]) for row in data]

# window_size = 1000
# moving_average = np.convolve(y_data, np.ones(window_size)/window_size, mode='valid')
# ma_filler = np.array([moving_average[0]] * (window_size-1))
# moving_variance = np.convolve((np.array(y_data) - np.concatenate([ma_filler, moving_average]))**2, np.ones(window_size)/window_size, mode='valid')

# # Compute the upper and lower bounds of the variance area
# upper_bound = moving_average + np.sqrt(moving_variance)
# lower_bound = moving_average - np.sqrt(moving_variance)

# # Plot the moving average of the episode returns over training, along with the variance area
# plt.plot(range(window_size-1, len(y_data)), moving_average)
# # plt.fill_between(range(window_size-1, len(y_data)), upper_bound, lower_bound, alpha=0.2)

# # plt.ylim([0,50])
# plt.xlabel('Episodes')
# plt.ylabel('Average Episode Return')
# plt.title('Average Episode Return over Training with Variance Area')
# plt.show()

''''''

# import pandas as pd

# # read in the first CSV file
# df1 = pd.read_csv('models/PPO_Agents/PPO_manual.csv')
# # extract the data for the first plot
# x1 = df1.iloc[:, 1]
# y1 = df1.iloc[:, 2]

# # read in the second CSV file
# df2 = pd.read_csv('models/PPO_Agents/PPO_with_SB3.csv')
# # extract the data for the second plot
# x2 = df2.iloc[:, 1]
# y2 = df2.iloc[:, 2]

# # plot the data
# plt.plot(x1, y1, label='Manual Implementation')
# plt.plot(x2, y2, label='Stable Baselines 3')
# plt.xlabel('steps')
# plt.ylabel('average total reward')
# plt.legend()
# plt.grid()
# plt.show()

'''
Plot Epsilon Decay
'''

# import matplotlib.pyplot as plt

# def exploration_decay(max_episodes, episode_num, initial_epsilon=1.0, final_epsilon=0.1, episode_learning_starts=500000):
#     if episode < episode_learning_starts:
#         exploration_rate = initial_epsilon
#     elif episode_num < 0.5*max_episodes:
#         exploration_rate = initial_epsilon - (0.1/(max_episodes/2 - episode_learning_starts))*(episode_num - episode_learning_starts)
#     elif 0.5*max_episodes <= episode_num < 0.75*max_episodes:
#         exploration_rate = 0.9 - (0.7/(0.25*max_episodes))*(episode_num-0.5*max_episodes)
#     else:
#         exploration_rate = 0.2 - (0.1/(0.25*max_episodes))*(episode_num-0.75*max_episodes)
#     return exploration_rate

# max_episodes = 2000000
# episodes = []
# epsilons = []


# for episode in range(max_episodes):

#     episodes.append(episode)
#     epsilons.append(exploration_decay(max_episodes, episode))

# plt.plot(episodes, epsilons)
# plt.show()

'''
Plot for new Parking and Final Punctuality
'''

# import matplotlib.pyplot as plt

# errors = []
# rewards = []
# rewards_old = []

# for error in np.arange(0.0, 200.1, 0.1):

#     errors.append(error)
#     # rewards.append(1 - (error/60)**0.63)
#     r = max(1 - (error/60) ** 0.63, -1.0)
#     rewards.append(r)

# plt.plot(errors, rewards)
# plt.ylabel('reward')
# plt.xlabel('punctuality error')
# plt.grid(color='lightgrey')

# plt.show()
