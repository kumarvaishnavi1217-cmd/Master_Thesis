
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import gym
import gym_train
import numpy as np
 #import gym_train.gym_train  # ensures env is registered
from wp1.state_spec import ALL_FEATURES

def main():
    env = gym.make("train-v0")
    obs = env.reset()

    state_vec = np.array(obs["1"]).reshape(-1)
    speed_limits = np.array(obs["speedlimits"])  # shape (27,2)

    expected = len(ALL_FEATURES)
    actual = len(state_vec) + speed_limits.size  # 10 + (27*2) = 64

    print("Expected features:", expected)
    print("Actual features:", actual)
    print("obs['1'] len:", len(state_vec))
    print("obs['speedlimits'] shape:", speed_limits.shape)

    assert actual == expected, "State mismatch between env and state_spec!"
    print("✅ State definition matches environment (64 features).")

    env.close()

if __name__ == "__main__":
    main()
