import gym

env = gym.make("gym_train:train-v0")
obs = env.reset()
obs, reward, done, info = env.step(5)  # neutral action

print("reward:", reward)
print("info keys:", list(info.keys()))
print("r_safety:", info["r_safety"], "r_energy:", info["r_energy"], "r_comfort:", info["r_comfort"])
print("termination_reason:", info["termination_reason"])

env.close()
