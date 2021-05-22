from .fourrooms_water import *
import numpy as np
import random
import pickle
import bz2

train_data_name = 'data_train'
test_data_name = 'data_test'
max_coins = 20
max_waters = 20
max_epilen = 100
trans_per_model = 5000
num_models = 100
num_train_models_of10 = 7
water_choices = ['pass', 'block', 'left', 'right', 'forward']
coin_choices = ['pass', 'left', 'right', 'forward']
action_choices = ['normal', 'left', 'right', 'inverse']
extra_step_choices = ['stay', 'left', 'right', 'forward']


def random_model():
    Model = dict()
    Model['water'] = np.random.choice(water_choices)
    Model['coin'] = np.random.choice(coin_choices)
    Model['action'] = np.random.choice(action_choices)
    # Model['extra step'] = np.random.choice(extra_step_choices)
    Model['extra step'] = 'stay'
    return Model


def model_to_onehot(Model: dict):
    water_onehot = [0] * len(water_choices)
    coin_onehot = [0] * len(coin_choices)
    action_onehot = [0] * len(action_choices)
    extra_step_onehot = [0] * len(extra_step_choices)
    water_onehot[water_choices.index(Model['water'])] = 1
    coin_onehot[coin_choices.index(Model['coin'])] = 1
    action_onehot[action_choices.index(Model['action'])] = 1
    extra_step_onehot[extra_step_choices.index(Model['extra step'])] = 1
    return np.concatenate([water_onehot, coin_onehot, action_onehot, extra_step_onehot]).astype(int)


def to_vector(state: FourroomsWaterState):
    # return list[pos_n, goal_n, coin_list, water_list]
    # coin_list and water_list with maximum element, -1 for empty, in descending order
    state_list = [state.position_n, state.goal_n]
    coin_list = []
    for coin, coin_state in state.coin_dict.items():
        if coin_state[1]:
            coin_list.append(coin)
    while len(coin_list) < max_coins:
        coin_list.append(-1)
    coin_list.sort(reverse=True)
    water_list = state.water_list.tolist()
    while len(water_list) < max_waters:
        water_list.append(-1)
    water_list.sort(reverse=True)
    state_list.extend(coin_list)
    state_list.extend(water_list)
    return tuple(state_list)


def collect_from_model(Model, num_trans, file):
    descr = FourroomsWater.todescr(Model)
    pickle.dump(descr, file)
    num_coins = random.randint(0, max_coins)
    num_waters = random.randint(0, max_waters)
    env = FourroomsWaterNorender(Model=Model, max_epilen=max_epilen, num_coins=num_coins, num_waters=num_waters,
                                 seed=None)
    current_trans = 0
    while current_trans < num_trans:
        if env.state.done:
            num_coins = random.randint(0, max_coins)
            num_waters = random.randint(0, max_waters)
            env = FourroomsWaterNorender(Model=Model, max_epilen=max_epilen, num_coins=num_coins, num_waters=num_waters,
                                         seed=None)
        current_state = to_vector(env.state)
        action = random.randint(0, env.action_space.n - 1)
        env.step(action)
        next_state = to_vector(env.state)
        trans = (current_state, action, next_state)
        pickle.dump(trans, file)
        current_trans += 1


def collect_data():
    # Collecting data, 7 for train, 3 for test by turns
    # No overlaps between train and test
    train_f = bz2.BZ2File(train_data_name, 'w')
    test_f = bz2.BZ2File(test_data_name, 'w')
    train_Model = dict()
    test_Model = dict()
    current_models = 0
    train_flag = 1
    test_flag = 0
    num_train_of10 = 0
    num_test_of10 = 0
    while current_models < num_models:
        if train_flag:
            while True:
                Model = random_model()
                if not test_Model.get(FourroomsWater.todescr(Model), False):
                    break
            train_Model[FourroomsWater.todescr(Model)] = True
            collect_from_model(Model, trans_per_model, train_f)
            num_train_of10 += 1
        if test_flag:
            while True:
                Model = random_model()
                if not train_Model.get(FourroomsWater.todescr(Model), False):
                    break
            test_Model[FourroomsWater.todescr(Model)] = True
            collect_from_model(Model, trans_per_model, test_f)
            num_test_of10 += 1
        if num_train_of10 >= num_train_models_of10:
            train_flag = 0
            test_flag = 1
            num_train_of10 = 0
        if num_test_of10 >= 10 - num_train_models_of10:
            train_flag = 1
            test_flag = 0
            num_test_of10 = 0
        current_models += 1


def desc2dict(desc: tuple):
    desc_dict = dict()
    for item in desc:
        k, v = item.split(':')
        desc_dict[k] = v[1:]
    return desc_dict


def generate_models(num_models=50):
    train_Model = dict()
    test_Model = dict()
    current_models = 0
    train_flag = 1
    test_flag = 0
    num_train_of10 = 0
    num_test_of10 = 0
    while current_models < num_models:
        if train_flag:
            while True:
                Model = random_model()
                if not test_Model.get(FourroomsWater.todescr(Model), False):
                    break
            train_Model[FourroomsWater.todescr(Model)] = True
            num_train_of10 += 1
        if test_flag:
            while True:
                Model = random_model()
                if not train_Model.get(FourroomsWater.todescr(Model), False):
                    break
            test_Model[FourroomsWater.todescr(Model)] = True
            num_test_of10 += 1
        if num_train_of10 >= num_train_models_of10:
            train_flag = 0
            test_flag = 1
            num_train_of10 = 0
        if num_test_of10 >= 10 - num_train_models_of10:
            train_flag = 1
            test_flag = 0
            num_test_of10 = 0
        current_models += 1
    return list(train_Model.keys()), list(test_Model.keys())


if __name__ == '__main__':
    train_Model, test_Model = generate_models()
    print(len(train_Model))
    # collect_data()
