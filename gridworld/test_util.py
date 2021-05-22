'''
Test scripts:
- check_render(env):check rendering
- check_run(env):random run
'''
import cv2
import os
def check_render(env):
    #check if it renders correctly
    if not os.path.exists('test_render'):
        os.mkdir('test_render')
    path='test_render'

    env.reset()
    cv2.imwrite(os.path.join(path,'test0.jpg'),env.render())
    env.step(0)
    cv2.imwrite(os.path.join(path,'test1.jpg'),env.render())
    env.step(1)
    cv2.imwrite(os.path.join(path,'test2.jpg'),env.render())
    env.step(2)
    cv2.imwrite(os.path.join(path,'test3.jpg'),env.render())
    env.step(3)
    cv2.imwrite(os.path.join(path,'test4.jpg'),env.render())

def check_run(env):
    #Run a few episodes
    reward_list=[]
    for i in range(1000):
        obs,reward,done,_=env.step(env.action_space.sample())
        reward_list.append(reward)
        if done:
            env.reset()
            reward_list=[]

