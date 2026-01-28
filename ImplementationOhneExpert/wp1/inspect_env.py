import gym
import numpy as np

def main():
    env = gym.make("gym_train:train-v0")
    obs = env.reset()
    named_obs = env.get_named_observation(obs)

    print("\nNamed observation:")
    for k, v in list(named_obs.items())[:10]:
        print(f"{k:30s} {v:.3f}")
    print("Obs type:", type(obs))
    print("Obs keys:", list(obs.keys()))

    if "1" in obs:
        print("obs['1'] shape:", np.array(obs["1"]).shape)
        print("obs['1'] sample:", obs["1"][:5])
    if "speedlimits" in obs:
        sl = np.array(obs["speedlimits"])
        print("obs['speedlimits'] shape:", sl.shape)
        print("obs['speedlimits'] sample (first 3):", sl[:3] if sl.ndim > 0 else sl)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    env.close()

if __name__ == "__main__":
    main()
