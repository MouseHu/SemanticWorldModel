"""Fourrooms Game with Coins 


class:
    + FourroomsCoinState and FourroomsCoin are based on FourroomsBase,
    you can inherit them for extension.
    + FourroomsCoinNorender support rendering.
    + FourroomsCoinBackgroundNoise is an extension example.

------
Design principles:
- seperate state,observation and rendering
- Each heirarchy should be a complete game.
- easy saving and loading
- gym interface

------
Some possible extensions:
- To change render colors or add background noises,rewrite render functions.
NOTE:ANY change of layout size should accompany a redefination of observation_space or obs_height and obs width.
- To add enemys,inherit FourroomsCoin.
- To change game layout,rewrite init_layout in fourrooms.py.
- This file includes extension examples: FourroomsCoinBackgroundNoise and FourroomsCoinRandomNoise.
- To add randomness, change self.random.
"""

import gym
import time
from gym import error, core, spaces
from gym.envs.registration import register
import random
import numpy as np
from .fourrooms import *
from .wrappers import ImageInputWarpper
from copy import deepcopy
import abc
import cv2
import time
from .test_util import *


class FourroomsCoinState(FourroomsBaseState):
    """
    The class that contains all information needed for restoring a FourroomsCoin game.
    The saving and restoring game must be of the same class and parameters.
    ···
    Attributes:
    position_n: int
        The numeralized position of agent.
    current_step: int
    goal_n:int
        The numeralized position of goal.
    done: bool
    coin_dict: dict int->(int,bool)
        coin->(value,if_exist)
    ...
    
    """

    def __init__(self, position_n: int, current_steps: int, goal_n: int, done: bool, num_pos: int, \
                 coin_dict: dict, num_coins, cum_reward: list):
        self.position_n = position_n
        self.current_steps = current_steps
        self.goal_n = goal_n
        self.done = done
        self.num_pos = num_pos
        self.coin_dict = coin_dict
        self.num_coins = num_coins
        self.cum_reward = cum_reward

    def __init__(self, base: FourroomsBaseState, coin_dict, num_coins, cum_reward):
        self.position_n = base.position_n
        self.current_steps = base.current_steps
        self.goal_n = base.goal_n
        self.done = base.done
        self.num_pos = base.num_pos
        self.coin_dict = coin_dict
        self.num_coins = num_coins
        self.cum_reward = cum_reward

    def coined_state(self):
        num_coins = 0
        for k, v in self.coin_dict.items():
            num_coins += 1
        value_list = [(v[0] if v[1] else 0) for v in self.coin_dict.values()]
        multiplier = np.dot(value_list, [2 ** i for i in range(num_coins)])
        return multiplier * self.num_pos + self.position_n

    def to_obs(self) -> np.array:
        return np.array(self.coined_state())

    def to_tuple(self) -> tuple:
        # convert dict to tuple of tuples
        l = []
        for k, v in self.coin_dict.items():
            l.append((k, v[0], v[1]))
        return self.position_n, self.current_steps, self.goal_n, self.done, tuple(l)


class FourroomsCoin(FourroomsNorender):
    """Fourroom game with agent,goal and coins that inherits gym.Env.

    This class should not render.
    """

    def __init__(self, max_epilen=100, goal=None, num_coins=3, seed=0):
        # 这里为了兼容留下了random coin，实际上没有用

        super(FourroomsCoin, self).__init__(max_epilen, goal, seed=seed)
        self._seed = seed
        self.num_coins = num_coins
        assert self.num_pos > (num_coins + 5), "too many coins"
        self.observation_space = spaces.Discrete(self.num_pos * (2 ** num_coins))
        coin_list = np.random.choice(self.init_states, num_coins, replace=False)

        if num_coins > 0:
            self.have_coin = True
        # You can change the value of coin here
        coin_dict = {coin: (1, True) for coin in coin_list}
        # random encode
        super().reset()

        self.state = FourroomsCoinState(self.state, coin_dict, num_coins, cum_reward=[])

    def step(self, action):
        if self.state.done:
            raise Exception("Environment should be reseted")
        currentcell = self.tocell[self.state.position_n]
        possible_actions = list(range(self.action_space.n))
        try:
            nextcell = tuple(currentcell + self.directions[action])
            possible_actions.remove(action)
        except TypeError:
            nextcell = tuple(currentcell + self.directions[action[0]])
            possible_actions.remove(action[0])

        if np.random.uniform() < self.random:  # random or determined
            random_action = np.random.choice(possible_actions)
            nextcell = tuple(currentcell + self.directions[random_action])

        if not self.occupancy[nextcell]:
            currentcell = nextcell

        state = self.tostate[currentcell]
        self.state.position_n = state
        if state == self.state.goal_n:
            reward = 10
        elif self.state.coin_dict.get(state, (0, False))[1]:  # if find coin
            reward = self.state.coin_dict.get(state, 0)[0] * 10
            self.state.coin_dict[state] = (self.state.coin_dict[state][0], False)
        else:
            reward = -0.1
        self.state.cum_reward.append(reward)

        self.state.current_steps += 1
        self.state.done = (state == self.state.goal_n) or self.state.current_steps >= self.max_epilen

        info = {}
        if self.state.done:
            info = {'episode': {'r': np.sum(self.state.cum_reward), 'l': self.state.current_steps}}
            # print(np.sum(self.state.cum_reward))
            self.state.cum_reward = []

        return self.state.to_obs(), reward, self.state.done, info

    def reset(self):
        super().reset()
        self.state = FourroomsCoinState(self.state, {}, self.num_coins, [])
        init_states = deepcopy(self.init_states)
        if self.state.goal_n in init_states:
            init_states.remove(self.state.goal_n)
        if self.state.position_n in init_states:
            init_states.remove(self.state.position_n)
        #np.random.seed(self._seed)
        coin_list = np.random.choice(init_states, self.num_coins, replace=False)
        #print("resetting coins")
        #print(coin_list)
        coin_dict = {coin: (1, True) for coin in coin_list}

        self.state.coin_dict = coin_dict

        return self.state.to_obs()

    @abc.abstractmethod
    def render(self):
        pass


class FourroomsCoinNorender(FourroomsCoin):
    def __init__(self, max_epilen=100, goal=None, num_coins=3, seed=0):
        super().__init__(max_epilen=max_epilen, goal=goal, num_coins=num_coins, seed=seed)

    def render(self, mode=0):
        blocks = []
        return self.render_coin_blocks(blocks)

    def render_coin_blocks(self, blocks=[]):
        """
        You must explicitly pass blocks
        """
        for coin, count in self.state.coin_dict.items():
            x, y = self.tocell[coin]
            if count[1]:  # exist
                blocks.append(self.make_block(x, y, (1, 1, 0)))
        blocks.extend(self.make_basic_blocks())

        arr = self.render_with_blocks(self.origin_background, blocks)
        return arr

    def render_huge(self):
        tmp = self.block_size
        self.block_size = 50
        self.obs_height = 50 * self.Row
        self.obs_width = 50 * self.Col

        self.init_background()
        arr = self.render()
        self.block_size = tmp
        self.obs_height = tmp * self.Row
        self.obs_width = tmp * self.Col
        self.init_background()
        return arr


# an extension example
class FourroomsCoinBackgroundNoise(FourroomsCoinNorender):
    """
    dynamic background noise
    """

    def __init__(self, max_epilen=400, obs_size=128, seed=0):
        super(FourroomsCoinBackgroundNoise, self).__init__(max_epilen, seed=seed)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size

        self.background = np.zeros((2, obs_size, obs_size, 3), dtype=np.int)
        self.background[0, :, :, 1] = 127  # red background
        self.background[1, :, :, 2] = 127  # blue background

    def render(self, mode=0):
        which_background = self.state.position_n % 2
        obs = deepcopy(self.background[which_background, ...])
        arr = super().render()
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs


class FourroomsCoinRandomNoise(FourroomsCoinNorender):
    """
    Random background noise
    """

    def __init__(self, max_epilen=100, obs_size=128, seed=0, num_colors=200, goal=77):
        super(FourroomsCoinRandomNoise, self).__init__(max_epilen=max_epilen, seed=seed)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size

        self.num_colors = num_colors
        self.color = np.random.randint(0, 255, (self.num_colors, 3))
        self.state_space_capacity = self.observation_space.n

    def render(self, mode=0):
        obs = np.tile(self.color[np.random.randint(0, self.num_colors)][np.newaxis, np.newaxis, :],
                      (self.obs_size, self.obs_size, 1))

        arr = super(FourroomsCoinRandomNoise, self).render()
        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr
        return obs.astype(np.uint8)


# random block that appears near the agent,but different from coin/goal/agent/wall
class FourroomsKidNoise(FourroomsCoinNorender):
    """
    FourroomsCoin Game with a kid randomly appears.
    """

    def __init__(self, max_epilen=100, obs_size=128, seed=int(time.time()) % 1024):
        super(FourroomsKidNoise, self).__init__(max_epilen, seed=seed)
        # self.background = np.zeros((2, obs_size, obs_size, 3),dtype=np.int)
        self.obs_size = obs_size
        self.obs_height = obs_size
        self.obs_width = obs_size

        self.background = (np.random.rand(3, obs_size, obs_size, 3) * 64).astype(np.int)
        self.background[0, :, :, 1] += 32  # distinguish background
        self.background[1, :, :, 2] += 32
        self.background[1, :, :, 0] += 32
        # kid

    def render(self, mode=0):
        which_background = self.state.position_n % 3
        obs = deepcopy(self.background[which_background, ...])
        # add kid
        avail_states = deepcopy(self.init_states)
        for k in self.state.coin_dict.keys():
            if k in avail_states:
                avail_states.remove(k)
        kid = np.random.choice(avail_states)
        x, y = self.tocell[kid]
        blocks = []
        blocks.append(self.make_block(x, y, (0, 0.5, 0.5)))
        #
        arr = self.render_coin_blocks(blocks)

        padding_height, padding_width = (obs.shape[0] - arr.shape[0]) // 2, (obs.shape[1] - arr.shape[1]) // 2
        obs[padding_height:padding_height + arr.shape[0], padding_width:padding_width + arr.shape[1], :] = arr - \
                                                                                                           obs[
                                                                                                           padding_height:padding_height +
                                                                                                                          arr.shape[
                                                                                                                              0],
                                                                                                           padding_width:padding_width +
                                                                                                                         arr.shape[
                                                                                                                             1],
                                                                                                           :]
        # background:[0,16]
        # obs:[0,255+16]

        new_obs = self.resize_obs(obs).astype(np.uint8)

        return new_obs

    def resize_obs(self, obs):
        """
        Resize observation array to [0,255]
        """
        # input:(obs_size,obs_size,3)
        minimum = np.min(obs)
        obs_zero_min = obs - minimum
        maximum = np.max(obs_zero_min)
        # -1 to make sure maximum point <255
        return ((obs_zero_min * 255) / maximum) - 1


if __name__ == '__main__':
    # basic test
    # env=ImageInputWarpper(FourroomsCoinNorender(seed=int(time.time())))
    env = ImageInputWarpper(FourroomsKidNoise())
    check_render(env)

    check_run(env)
    print("basic check finished")
    # stable-baseline test
    # check_env(env,warn=True)
