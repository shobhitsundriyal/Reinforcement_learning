import gym

env = gym.make('MountainCar-v0')
env.reset()

done = False

while not done:
    ''' action 0 ==> go  left
        action 1 ==> do nothing
        action 2 ==> go right
    '''
    action = 2
    new_state, reward, done, _ = env.step(action)
    env.render()

env.close()