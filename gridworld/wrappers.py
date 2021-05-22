import gym
from gym import core, spaces
import numpy as np
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        
class ImageInputWarpper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        screen_height = self.env.obs_height
        screen_width = self.env.obs_width
        self.observation_space = spaces.Box(low=0, high=255, shape=\
        (screen_height, screen_width,3), dtype=np.uint8)
        self.mean_obs = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        obs = self.env.render()
        info['s_tp1'] = state
        return obs.astype(np.uint8), reward, done, info

    def reset(self):
        self.env.reset()
        obs = self.env.render()
        return obs.astype(np.uint8)
