'''
Test scripts:
- check_env(env):imported from stable_baselines,check whether it follows gym interface
- I also try to train an agent with stable_baselines to check the difficulty and reasonability of game.
'''

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import ACER,A2C,ACKTR
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy


def stable_baseline_test(env_origin):
    env = make_vec_env(lambda: env_origin, n_envs=1)
    model = ACKTR('CnnPolicy',env_origin, verbose=1)
    model.learn(total_timesteps=2000000)
    print("Stable_baseline evaluation starts.....\n")
    #NOTE:evaluate_policy needs vec_env
    reward_mean,reward_std=evaluate_policy(model,env,n_eval_episodes=20,deterministic=False)

    print("mean reward:"+str(reward_mean)+'\n')
    print("reward std:"+str(reward_std)+'\n')

    print("custom evaluation begin\n")

    env=env_origin
    obs = env.reset()
    reward_list_total=[]
    epilen_list=[]
    reward_list=[]
    last_end=0
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward_list.append(rewards)
        if dones:
            obs=env.reset()
            epilen_list.append(i-last_end)
            last_end=i
            reward_list_total.append(np.sum(reward_list))
            reward_list=[]
            if i>900:
                break
    print("mean reward:{}\n".format(np.mean(reward_list_total)))
    print("mean epilen:{}\n".format(np.mean(epilen_list)))