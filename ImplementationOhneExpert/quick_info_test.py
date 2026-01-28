import gym_train.envs.train_env as te
print("Using file:", te.__file__)

from gym_train.envs.train_env import TrainEnv

env = TrainEnv()
obs = env.reset()

print("\nSTEP-BY-STEP (5 steps)")
for t in range(5):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    print(
        f"t={t:02d}  a={action:2d}  reward={reward: .3f}  "
        f"dist={info['distance_to_destination']: .1f}  "
        f"r_prog={info['r_progress']: .4f}  r_tw={info['r_timewaste']: .3f}  r_term={info['r_terminal']: .1f}  "
        f"dist={info['distance_to_destination']: .1f}"
        f"dist={info['distance_to_destination']: .1f}"
        f"speed={info['speed']: .2f}  reason={info['termination_reason']}"
    )

    if done:
        break

print("\nreward_components:", info["reward_components"])
