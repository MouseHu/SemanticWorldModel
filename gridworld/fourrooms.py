"""Basic Fourrooms Game 

This script contains a basic version of Fourrooms.

If you want to extend the game,please inherit FourroomsBaseState and FourroomsBase.

Some design principles,extension advice and test information can be seen in fourrooms_coin.py.

"""
import numpy as np
import gym
import time
from gym import error, spaces
from gym import core, spaces
from gym.envs.registration import register
import random
from copy import deepcopy
from .wrappers import ImageInputWarpper
import cv2
from .test_util import check_render, check_run


class FourroomsBaseState(object):
    """State of FourroomsBase

    The class that contains all information needed for restoring a game.
    The saving and restoring game must be of the same class and the same instance.
    This class is designed for FourroomsBase.
    ···
    Attributes:
    position_n: int
        The numeralized position of agent.
    current_step: int
    goal_n:int
        The numeralized position of goal.
    done: bool
        whether position_n==goal_n or current_steps>max_epilen
    num_pos:int
        number of positions in env,saved for convenience and It should not be changed
    """

    def __init__(self, position_n: int, current_steps: int, goal_n: int, done: bool, num_pos: int):
        self.position_n = position_n
        self.current_steps = current_steps
        self.goal_n = goal_n
        self.done = done
        self.num_pos = num_pos

    def to_obs(self) -> np.array:
        return np.array(self.position_n)

    def to_tuple(self):
        return (self.position_n, self.current_steps, self.goal_n, self.done)


class FourroomsBase(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    """Fourroom game with agent and goal that inherits gym.Env.

    ···
    Attributes:
    ------------
    occupancy: map
        from (x,y) to [0,1],check whether the position is blocked.
    num_pos: int 
        the number of non-blocked positions
    block_size: int
        length of a squared block measured in pixels 
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    tocell: dict (x,y)->state
        a map from position to state
    tostate: dict state->(x,y)
    init_states: list
        all non-blocked states to place objects,REMAIN CONSTANT
    max_epilen: int
        maximum episode length
    current_steps: int
        current step
    currentcell:(x,y)
        current cell
    unwrapped: self
        origin env
    state: FourroomsBaseState
        internal state
    observation: np.array
        part of the state
    open: bool
        whether game is running
    
    Methods: Following gym interface
    ---------
    step
    reset
    close
    seed
    ...
    """

    def __init__(self, max_epilen=100, goal=None, seed=0, random=False):
        """
        goal:None means random goal
        """
        self.have_coin = False
        self.random = 1 / 3 if random else 0
        self.seed(seed)
        self.init_layout()
        self.init_basic(max_epilen, goal)

    def init_layout(self):
        self.layout = """\
1111111111111
1     1     1
1     1     1
1           1
1     1     1
1     1     1
11 1111     1
1     111 111
1     1     1
1     1     1
1           1
1     1     1
1111111111111
"""
        self.block_size = 8
        self.occupancy = np.array(
            [np.array(list(map(lambda c: 1 if c == '1' else 0, line))) for line in self.layout.splitlines()])
        self.num_pos = int(np.sum(self.occupancy == 0))
        self.Row = np.shape(self.occupancy)[0]
        self.Col = np.shape(self.occupancy)[1]
        self.obs_height = self.block_size * self.Row
        self.obs_width = self.block_size * self.Col

    def init_basic(self, max_epilen, goal):
        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.num_pos)
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.rand_color = np.random.randint(0, 255, (200, 3))  # low,high,size
        self.tostate = {}

        statenum = 0
        # label states
        for i in range(len(self.occupancy)):
            for j in range(len(self.occupancy[0])):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}
        self.init_states = list(range(self.observation_space.n))
        self.dict = dict()

        self.max_epilen = max_epilen
        self.get_dict()

        self.reward_range = (0, 1)
        self.metadata = None
        self.allow_early_resets = True
        if goal != None and goal > (self.observation_space.n):
            raise ValueError("invalid goal position")
        self.goal = goal

    def get_dict(self):
        """
        Label positions of states
        TODO:add infomation(e.g. goal/agent) to positions
        """
        count = 0
        for i in range(self.Row):
            for j in range(self.Col):
                if self.occupancy[i, j] == 0:
                    # code
                    self.dict[count] = (i, j)
                    count += 1

    def empty_around(self, cell: tuple) -> list:
        """
        Find all available cells around the cell.
        """
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        """
        reset state,rechoose goal position if needed
        """
        self.open = True

        init_states = deepcopy(self.init_states)
        if self.goal == None:
            print("resetting goal")
            goal = np.random.choice(init_states)
            self.goal=goal
            print("goal",goal)
        else:
            # print("Not resetting goal")
            goal = self.goal
        try:
            init_states.remove(goal)
        except Exception as e:
            print(init_states)
            print(goal)
            print(e)
        #np.random.seed(int(time.time()))
        init_position = np.random.choice(init_states)
        #print(f'setting init:{init_position}')

        self.state = FourroomsBaseState(position_n=init_position, current_steps=0, goal_n=goal, done=False,
                                        num_pos=self.num_pos)

        return self.state.to_obs()

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.

        We consider a case in which rewards are zero on all state transitions.
        """
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

        position_n = self.tostate[currentcell]

        self.state.current_steps += 1
        self.state.done = (position_n == self.state.goal_n) or (self.state.current_steps >= self.max_epilen)
        info = {}
        if self.state.done:
            # print(self.current_step, state == self.goal, self.max_epilen)
            info = {'episode': {'r': 10 - self.state.current_steps * 0.1 if (position_n == self.state.goal_n) \
                else -self.state.current_steps * 0.1,
                                'l': self.state.current_steps}}

        self.state.position_n = position_n

        if self.state.position_n == self.state.goal_n:
            reward = 10
        else:
            reward = -0.1
        return self.state.to_obs(), reward, self.state.done, info

    def seed(self, seed):
        np.random.seed(seed)

    def close(self):
        pass

    def render(self):
        raise NotImplementedError()

    def inner_state(self):
        return self.state

    def load(self, state):
        self.state = state


class FourroomsNorender(FourroomsBase):
    """
    A rendered version.
    Image :(104,104,3)
    """

    def __init__(self, max_epilen=100, goal=77, seed=0):
        super().__init__(max_epilen, goal, seed)

        self.agent_color = np.random.rand(100, 3)
        self.init_background()
        # print(self.background.shape)

    def init_background(self):
        self.wall_blocks = self.make_wall_blocks()
        # render origin wall blocks to speed up rendering
        self.origin_background = self.render_with_blocks(
            255 * np.ones((self.obs_height, self.obs_width, 3), dtype=np.uint8),
            self.wall_blocks)

    def render(self, mode=0):
        """
        Render currentcell\walls\background,you can add blocks by parameter.
        Render mode is reserved for being compatible with gym interface.
        """
        # render agent
        blocks = (self.make_basic_blocks())

        arr = self.render_with_blocks(self.origin_background, blocks)

        return arr

    def make_basic_blocks(self):
        blocks = []
        currentcell = self.tocell[self.state.position_n]
        if currentcell[0] > 0:
            x, y = currentcell
            # blocks.append(self.make_block(x, y, self.agent_color[np.random.randint(100)]))
            blocks.append(self.make_block(x, y, (0, 0, 1)))
        # render goal
        if self.state.position_n != self.state.goal_n:
            x, y = self.tocell[self.state.goal_n]
            blocks.append(self.make_block(x, y, (1, 0, 0)))
        return blocks

    def render_with_blocks(self, background, blocks) -> np.array:
        background = np.copy(np.array(background))
        assert background.shape[-1] == len(background.shape) == 3, background.shape
        for block in blocks:
            v, color = block
            color = np.array(color).reshape(-1) * 255
            background[v[0][0]:v[2][0], v[0][1]:v[2][1], :] = color.astype(np.uint8)
        # assert background.shape[-1] == len(background.shape) == 3,background.shape
        return background

    def make_wall_blocks(self):

        blocks = []
        size = self.block_size
        for i, row in enumerate(self.occupancy):
            for j, o in enumerate(row):
                if o == 1:
                    v = [[i * size, j * size], [i * size, (j + 1) * size], [(i + 1) * size, (j + 1) * size],
                         [(i + 1) * size, (j) * size]]
                    color = (0, 0, 0)
                    geom = (v, color)
                    blocks.append(geom)
        return blocks

    def make_block(self, x, y, color):
        """
        color in [0,1]
        """
        size = self.block_size
        v = [[x * size, y * size], [x * size, (y + 1) * size], [(x + 1) * size, (y + 1) * size],
             [(x + 1) * size, y * size]]
        geom = (v, color)
        return geom

    def render_state(self, state):
        # a temp workaround
        tmp, self.state = self.state, state
        arr = self.render()
        self.state = tmp
        return arr

    def render_huge(self):
        # a util function for value iteration.
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


if __name__ == '__main__':
    # basic test
    env_origin = ImageInputWarpper(FourroomsNorender())
    check_render(env_origin)
    check_run(env_origin)

    # stable-baseline test
    # check_env(env_origin,warn=True)
