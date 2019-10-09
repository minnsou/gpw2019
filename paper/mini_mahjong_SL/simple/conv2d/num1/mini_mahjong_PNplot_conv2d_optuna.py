#!/usr/bin/env python
# coding: utf-8

#  **変数の意味**
# 
# - state : あがりに必要な枚数の手牌の状態
# 
# - hand : stateから一枚切った状態
# 
# - n : 用いる牌の種類数
# 
# - m : 手牌の枚数
# 
# - l : 同一牌の枚数(基本的に4で固定)
# 
# optunaとtrain_test_splitは使用

from collections import defaultdict
import itertools
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D
from sklearn.model_selection import train_test_split
import optuna
from optuna.integration import KerasPruningCallback
import time

random_seed = 34
np.random.seed(random_seed)

# ### 必要な関数の再定義


def is_valid(seq, l=4): # 生成された組み合わせが手牌として妥当かどうかを判断する関数　tuple(seq)の一つ一つが一つの状態(手牌)に対応している
    counts = defaultdict(lambda: 0)
    for i in range(0, len(seq)):
        if i + 1 < len(seq) and seq[i] > seq[i + 1]: # 前半の条件はiが一番最後以外は常に成立、後半の条件は昇順に整列するための条件
            return False
        counts[seq[i]] += 1
        if (counts[seq[i]] > l): return False # 牌の上限枚数を超えたらFalse
    return True

def number_state_slow(n,m,l): # 全ての手牌の組み合わせの数を出力する関数
    count = 0
    for seq in itertools.product(range(n), repeat = m): # 直積を作る関数, n=9 m=5 なら 9 ** 5 回繰り返す　
        if is_valid(seq,l):
            count += 1
            #print(list(seq))
    return count
    
def generate_all_l(n, m, l=4): # 全ての手牌の組み合わせをタプルで出力する関数
    gen_list = []
    for seq in itertools.product(range(n), repeat = m):
        if is_valid(seq, l):
            gen_list.append(seq)
    return gen_list

def states_to_hist(state_list, n): # 手牌(state)を、牌種ごとの枚数のリスト(長さn)に変換する関数
    hist_list = []
    for state in state_list:
        #print(state)
        ret = [0] * n # ret = [0,0,...,0]
        for c in state:
            ret[c] += 1
        hist_list.append(ret)
    return hist_list

def hand_to_prob_and_state(hand, state_nml, n, m, l=4): # ある手牌(hand)における、1枚ツモる時の遷移確率(prob)と手牌(state)のindexのタプルを出す関数
    ret = [l] * n  #  残り枚数を表すリスト
    for h in hand:
        ret[h] -= 1
    yama_sum = n * l - (m - 1)
    state_list = []
    for i in range(n):
        if ret[i] == 0: 
            continue
        prob = ret[i] / yama_sum # 遷移確率
        state = tuple(sorted(list(hand) + [i])) # 遷移後の手牌
        #print(state)
        state_index = state_nml.index(state) # 遷移後の手牌のindex
        #print(state_index)
        state_list.append((prob, state_index))
    return state_list

def state_to_hand(state): # ある手牌stateに遷移できるhandを出力する関数
    return list(set(tuple(state[:i] + state[i+1:]) for i in range(len(state)))) # i番目の要素を取り除く

def win_split_sub(hist, two, three, split_state, agari_list):
    if any(x < 0 for x in hist):
        return
    if two == 0 and three == 0:
        agari_list.append(tuple(split_state))
        return
    i = next(i for i, x in enumerate(hist) if x > 0) # histの中でx>０を満たす最小のindexを持ってくる
    next_hist = [x - 2 if i == j else x for j, x in enumerate(hist)]
    if two > 0 and hist[i] == 2: # 雀頭
        win_split_sub(next_hist, two - 1, three, split_state + [(i, i)], agari_list)
    next_hist = [x - 3 if i == j else x for j, x in enumerate(hist)]
    if three > 0 and hist[i] == 3: # 刻子
        win_split_sub(next_hist, two, three - 1, split_state + [(i, i, i)], agari_list)
    next_hist = [x -1 if i <= j <= i + 2 else x for j, x in enumerate(hist)]
    if three > 0 and i + 2 < len(hist): # 順子
        win_split_sub(next_hist, two, three - 1, split_state + [(i, i+1, i+2)], agari_list)
    return 
    
def win_split_main(hist):
    n_two = 1 if sum(hist) % 3 == 2 else 0
    n_three = sum(hist) // 3
    agari_list = []
    win_split_sub(hist, n_two, n_three, [], agari_list)
    if len(agari_list) == 0:
        return (False, set())
    else:
        return (True, agari_list)

# print(win_split_main([1, 1, 2, 2, 2]))
# print(win_split_main([2, 3, 3, 3, 3]))
# print(win_split_main([1, 1, 3, 3, 3]))
# print(win_split_main([0, 4, 4, 4, 2, 0]))
    
# def is_tanyao(state):
#     for hai in state:
#         if hai == 0 or hai == 8:
#             return False
#     return True

# def is_chanta(split_state):
#     state_value = True
#     for block in split_state:
#         if 0 in block or 8 in block:
#             continue
#         else:
#             state_value = False
#             break
#     return state_value

# def is_toitoi(split_state):
#     state_value = True
#     for block in split_state:
#         if len(block) == 2: # 雀頭
#             continue
#         else:  # 面子
#             if block[0] != block[1]:
#                 state_value = False
#                 break
#     return state_value
    
# def is_ipeko(split_state):
#     for block in split_state:
#         if len(block) == 2:
#             continue
#         if block[0] != block[1]:
#             temp = list(split_state)
#             temp.remove(block)
#             if block in temp:
#                 return True
#     return False

# def is_pinhu(split_state):
#     return False

def value_iteration(n, m, l, gamma):
    state_nml = generate_all_l(n, m, l)
    hand_nml = generate_all_l(n, m-1, l)
    hist_nml = states_to_hist(state_nml, n)
    is_win_nml, split_state_list_nml = [], []
    for i, hist in enumerate(hist_nml):
        val, split = win_split_main(hist)
        is_win_nml.append(val)
        split_state_list_nml.append(split)
    #print(is_win_nml)
    #print(split_state_set_nml)
    #is_win_nml = [is_win_main(hist) for hist in hist_nml]
    h2ps_nml = [hand_to_prob_and_state(hand, state_nml, n, m, l) for hand in hand_nml]
    s2h_nml = [[hand_nml.index(hand) for hand in state_to_hand(state)] for state in state_nml]
    value_hand = [0] * len(hand_nml)
    n_hand = len(hand_nml)
    value_state = [1 if is_win_nml[i] else 0 for i in range(len(state_nml))] # あがっていればvalueは1、いなければ0
    # value_state = [2 * value_state[i] if is_tanyao(state) else value_state[i] for i, state in enumerate(state_nml)] # tannyao
    # # 役判定(断么九以外)
    # for i, split_state_list in enumerate(split_state_list_nml):
    #     if len(split_state_list) == 0:
    #         continue
    #     elif len(split_state_list) == 1:
    #         if is_chanta(split_state_list[0]):
    #             value_state[i] *= 2
    #         if is_toitoi(split_state_list[0]):
    #             value_state[i] *= 2
    #         if is_ipeko(split_state_list[0]):
    #             value_state[i] *= 2
    #     else:
    #         max_state_value = value_state[i]
    #         for split_state in split_state_list:
    #             temp_state_value = value_state[i]
    #             if is_chanta(split_state):
    #                 temp_state_value *= 2
    #             if is_toitoi(split_state):
    #                 temp_state_value *= 2
    #             if is_ipeko(split_state):
    #                 temp_state_value *= 2                 

    #             if temp_state_value > max_state_value:
    #                 max_state_value = temp_state_value
    #         value_state[i] = max_state_value
    n_state = len(state_nml)
    theta = 1e-6
    while True:
        print('iteration')
        delta = 0
        for i in range(n_hand):
            old_v = value_hand[i]
            value_hand[i] = sum(p * value_state[n] for (p, n) in h2ps_nml[i])
            delta = max(delta, abs(old_v - value_hand[i]))
        if delta < theta: break
        for i in range(n_state):
            if is_win_nml[i]: continue
            value_state[i] = max(gamma * value_hand[n] for n in s2h_nml[i])
    return value_hand # 各valueのhandをリストにして返す

# 手牌(state)を、牌種ごとの枚数のリスト(長さn)に変換する関数
def state_to_hist(state, n):
    hist = [0] * n # hist = [0,0,...,0]
    for c in state:
        hist[c] += 1
    return hist

# stateとその時にvalueが最大となる捨て牌のタプルを入れたリスト max_value_discard_list = [((0, 0, 0, 0, 1), {0}), ((0, 0, 0, 0, 2), {0}), ... ,((7, 8, 8, 8, 8), {8})]
# state_nmlのうち、あがり形を抜いたもの discard_state_nml = [(0, 0, 0, 0, 1), (0, 0, 0, 0, 2), ..., (7, 8, 8, 8, 8)]
def states_to_max_value_list(state_nml, hand_nml, value_hand_nml, n, m, l=4):
    max_value_list = []
    discard_state_nml = []
    hist_nml = states_to_hist(state_nml, n)
    for i, hist in enumerate(hist_nml):        
        if win_split_main(hist)[0]:
            continue # あがっているstateの時は何も入れない
        else:
            max_value = 0
            max_p = []
            for j in range(m):
                state = state_nml[i]
                hand = state[:j] + state[j+1:]
                ind = hand_nml.index(tuple(hand))
                hand_val = value_hand_nml[ind]
                if max_value < hand_val:
                    max_p = {state[j]}
                    max_value = hand_val
                elif round(max_value, 5) == round(hand_val, 5): # 小数点以下5桁まで同じなら同じとみなす
                    max_p.add(state[j])
            discard_state_nml.append(state_nml[i])
            max_value_list.append(tuple((state_nml[i], max_p)))
    return max_value_list, discard_state_nml # 正直discard_hist_nmlを出す方が早い

# 各stateにおいて、出力してほしい捨て牌の確率分布を出力する
def discard_ans_prob_vector(max_value_discard_list, n, m, l):
    discard_vector = []
    for i, discard in max_value_discard_list:
        v = [0] * n
        num = len(discard)
        for p in discard:
            v[p] = 1 / num # 答えの数で割った値を教師とする  こうしないと学習がうまくいかない?
            #v[p] = 1
        discard_vector.append(v)
    return discard_vector


# one_hot化する関数たち
def one_hot_vector1(hands, n): # 手牌の中の牌一つ一つをone-hotにした(手牌１つがn * m-1の行列に対応)
    results = np.zeros((len(hands), n, len(hands[0])))
    for i in range(len(hands)):
        for j, hand_i in enumerate(hands[i]):
            results[i][hand_i][j] = 1
    return results

def one_hot_vector2(hists, n, l=4): # histをそのままone-hotにした(手牌１つがn * l + 1の行列に対応)
    results = np.zeros((len(hists), n, l + 1))
    for i in range(len(hists)):
        for j, hist_i in enumerate(hists[i]):
            results[i][j][hist_i] = 1
    return results

def one_hot_vector3(hists, n, l=4): # 上に近いけど、持ってる枚数より小さい数も1で埋めた(手牌１つがn * lの行列に対応)
    results = np.zeros((len(hists), n, l))
    for i in range(len(hists)):
        for j, hist_i in enumerate(hists[i]):
            if hist_i == 0:
                continue
            else:
                results[i][j][:hist_i] = 1
    return results

# discard_state_nmlをone_hot化する関数
def one_hot(discard_state_nml, num):
    if num == 1:
        return one_hot_vector1(discard_state_nml, n)
    elif num == 2:
        discard_hist_nml = states_to_hist(discard_state_nml, n)
        return one_hot_vector2(discard_hist_nml, n, l)
    else:
        discard_hist_nml = states_to_hist(discard_state_nml, n)
        return one_hot_vector3(discard_hist_nml, n, l)


# ### 捨て牌ベクトルの作成

n = 9
m = 8
l = 4

value_hand_nml = value_iteration(n, m, l, 0.9)
state_nml = generate_all_l(n, m, l)
hand_nml = generate_all_l(n, m - 1, l)
#print(len(hand_nml))
#print(hand_nml)
#print(value_hand_nml)


max_value_discard_list, discard_state_nml = states_to_max_value_list(state_nml, hand_nml, value_hand_nml, n, m, l)
#for i in max_value_discard_list: print(i) 
#discard_hist_nml = states_to_hist(discard_state_nml, n)

#one_hot_discard_state_nml1 = one_hot_vector1(discard_state_nml, n)
#one_hot_discard_state_nml2 = one_hot_vector2(discard_hist_nml, n, l)
#one_hot_discard_state_nml3 = one_hot_vector3(discard_hist_nml, n, l)
discard_ans_vector_nml = np.array(discard_ans_prob_vector(max_value_discard_list, n, m, l))
#print(discard_ans_vector_nml)


# ### Policy networkの作成


def train_policy_network(num, input_shape, n_study, EPOCHS):
    start_time = time.time()

    def create_model(trial):
        # We optimize the number of layers, hidden units and dropout in each layer and
        # the learning rate of RMSProp optimizer.

        # We define our myconv2d
        model = Sequential()
        model.add(layers.InputLayer(input_shape=input_shape))
        n_conv2d_layers = trial.suggest_int('n_conv2d_layers', 1, 4)
        for i in range(n_conv2d_layers):
            num_filters = trial.suggest_int('n_filters_|{}'.format(i), 8, 64)
            model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
        model.add(Flatten())
        n_layers = trial.suggest_int('n_layers', 1, 2)
        for i in range(n_layers):
            num_hidden = int(trial.suggest_loguniform('n_units_l{}'.format(i), 50, 1000))
            model.add(Dense(num_hidden, activation='relu'))
            #dropout = trial.suggest_uniform('dropout_l{}'.format(i), 0.2, 0.5)
            #model.add(Dropout(rate=dropout))
        model.add(Dense(n, activation='softmax'))

        # We compile our model with a sampled learning rate.
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.RMSprop(lr=lr),
                      metrics=['accuracy'])

        return model


    def objective(trial):
        # Clear clutter form previous session graphs.
        keras.backend.clear_session()
        
        discard_state_train, discard_state_test, discard_ans_vector_train, discard_ans_vector_test = train_test_split(discard_state_nml, discard_ans_vector_nml, test_size=0.25)
        one_hot_discard_state_train = one_hot(discard_state_train, num).reshape(len(discard_state_train), input_shape[0], input_shape[1], 1)
        one_hot_discard_state_test = one_hot(discard_state_test, num).reshape(len(discard_state_test), input_shape[0], input_shape[1], 1)
        # Generate our trial model.
        model = create_model(trial)

        # Fit the model on the training data.
        # The KerasPruningCallback checks for pruning condition every epoch.
        model.fit(one_hot_discard_state_train, discard_ans_vector_train, epochs=EPOCHS, callbacks=[KerasPruningCallback(trial, 'val_acc')], validation_data=(one_hot_discard_state_test, discard_ans_vector_test), verbose=0)

        # Evaluate the model accuracy on the test set.
        score = model.evaluate(one_hot_discard_state_test, discard_ans_vector_test, verbose=0)
        return score[1]
    
    for i in range(n_study):
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=200)
        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
        print('Study statistics: ')
        print('  Number of pruned trials: ', len(pruned_trials))
        print('  Number of complete trials: ', len(complete_trials))

        print('Best trial:')
        trial = study.best_trial
        print('  Value: ', trial.value)

        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
        print(round(time.time() - start_time), 'sec\n')    



num = 1
input_shape = (n, m, 1)
n_study = 1
EPOCHS = 1000
train_policy_network(num, input_shape, n_study, EPOCHS)


num = 2
input_shape = (n, l+1, 1)
n_study = 1
EPOCHS = 1000
#train_policy_network(num, input_shape, n_study, EPOCHS)


num = 3
input_shape = (n, l, 1)
n_trials = 1
EPOCHS = 1000
#train_policy_network(num, input_shape, n_study, EPOCHS)

