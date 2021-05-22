'''
Value iteration can provide knowledge about the game.
Only implemented for non-coin and one-coin case.
'''

import numpy as np
import pickle
import os
from copy import deepcopy
from fourrooms import FourroomsNorender
from fourrooms_coin import FourroomsCoinNorender
from wrappers import ImageInputWarpper
import cv2

def value_iteration(env, gamma=0.95, buffer_size=2000, showQ=True):
    """
    Value iteration calculates value function of each state.
    """
    env.reset()
    #for showing Q
    if not os.path.exists('value_iter'):
        os.mkdir('value_iter')
    path='value_iter'
    if showQ:
        img=env.render_huge()
    cv2.imwrite(os.path.join(path,'origin.jpg'),env.render())
    state_zero=env.state
    
    values={}
    transition={}
    rewards={}
    dones={}
    Q={}
    #fix goal_n,find all possible states that have the same goal，current_step is constantly 0
    all_states=[]
    possible_positions=list(range(state_zero.num_pos))
    possible_positions.remove(state_zero.goal_n)

    if not env.have_coin:
        #for env without coin
        for i in possible_positions:
            state_zero.position_n=i
            all_states.append(deepcopy(state_zero))
    else:
        for i in possible_positions:
            state_zero.position_n=i
            if len(list(state_zero.coin_dict.items()))==1:
                for k,v in state_zero.coin_dict.items():
                    if state_zero.position_n!=k:
                        state_zero.coin_dict[k]=(state_zero.coin_dict[k][0],True)
                        all_states.append(deepcopy(state_zero))

                    state_zero.coin_dict[k]=(state_zero.coin_dict[k][0],False)
                    all_states.append(deepcopy(state_zero))
            #FIXME
            else:
                raise NotImplementedError("value iteration for more than one coin is not implemented{}")
    
    #cover all states and transitions
    for state in all_states:
        state_t=state.to_tuple()
        
        values[state_t]=0
        rewards[state_t]=[0]*(env.action_space.n)
        transition[state_t]=[0]*(env.action_space.n)
        dones[state_t]=[0]*(env.action_space.n)
        
        for a in range(env.action_space.n):
            env.load(deepcopy(state))
            if env.state.done:
                print((env.state is state))
                print("what happened?")
                print(state_t)
            
            
            obs_tp1, reward, done, info = env.step(a)
            #assume infinite horizon 
            env.state.current_steps=0
            transition[state_t][a]=deepcopy(env.state)
            rewards[state_t][a]=reward
            dones[state_t][a]=done

    #ieratively calc Q value
    for _ in range((300)):
        for s in all_states:
            Q[s.to_tuple()]=[0]*4
            # q = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                Q[s.to_tuple()][a] = rewards[s.to_tuple()][a]
                if not dones[s.to_tuple()][a]:
                    Q[s.to_tuple()][a] += gamma * values[transition[s.to_tuple()][a].to_tuple()]

                Q[s.to_tuple()][a]=round(Q[s.to_tuple()][a],3)
            values[s.to_tuple()] = np.max(Q[s.to_tuple()])
    #print Q in screen
    if showQ:
        #FIXME:only correct when there is 1 coin
        
        for s in all_states:
            font = cv2.FONT_HERSHEY_PLAIN
            value=values[s.to_tuple()]
            cell=env.tocell[s.position_n]
            rel_pos=40 if list(s.coin_dict.items())[0][1][1] else 10
            cv2.putText(img, str(value),(cell[1]*50+10,cell[0]*50+rel_pos),font,0.7,(0,0,0),1)

        cv2.imwrite(os.path.join(path,'value_iteration.jpg'),img)
    #print(Q)

    print("value iteration finished")

if __name__=='__main__':
    env=ImageInputWarpper(FourroomsCoinNorender())
    value_iteration(env)