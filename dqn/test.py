import gym

env = gym.make('BreakoutDeterministic-v0')
env.reset()

while True:
    env.render()
    # Random action
    action = env.action_space.sample()
    next_state, reward, is_done, info = env.step(action)

    print("Reward: {}, is_done: {}".format(reward, is_done))

    if is_done:
        env.reset()