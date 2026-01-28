from gym.envs.registration import register

register(
    id='train-v0',
    entry_point='gym_train.envs:TrainEnv',
)