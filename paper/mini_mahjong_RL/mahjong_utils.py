import numpy as np

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

def is_tanyao(split_state):
    for block in split_state:
        if 0 in block or 8 in block:
            return False
    return True

def is_chanta(split_state):
    state_value = True
    for block in split_state:
        if 0 in block or 8 in block:
            continue
        else:
            state_value = False
            break
    return state_value

def is_toitoi(split_state):
    state_value = True
    for block in split_state:
        if len(block) == 2: # 雀頭
            continue
        else:  # 面子
            if block[0] != block[1]:
                state_value = False
                break
    return state_value

def is_ipeko(split_state):
    for block in split_state:
        if len(block) == 2:
            continue
        if block[0] != block[1]:
            temp = list(split_state)
            temp.remove(block)
            if block in temp:
                return True
    return False

def yaku_point(split_state):
    point = 1
    if is_tanyao(split_state):
        point *= 2
    if is_toitoi(split_state):
        point *= 2
    if is_ipeko(split_state):
        point *= 2
    if is_chanta(split_state):
        point *= 2
    return point

# 手牌(state)を、牌種ごとの枚数のリスト(長さn)に変換する関数
def state_to_hist(state, n):
    hist = [0] * n # hist = [0,0,...,0]
    for c in state:
        hist[c] += 1
    return hist

def hist2onehot(hand_mode, num_hand, kind_tile, NUM_SAME_TILE, hand):
    if hand_mode == 1: # 牌の1つ1つをone-hotへ
        r = np.zeros(shape=(num_hand, kind_tile), dtype=np.float32)
        num = 0
        for idx, h in enumerate(hand):
            for i in range(h):
                r[num][idx] = 1.0
                num += 1
    elif hand_mode == 2: # hist形式の手牌をone-hotへ
        r = np.zeros(shape=(kind_tile, NUM_SAME_TILE + 1), dtype=np.float32)
        for i in range(kind_tile):
            r[i][hand[i]] = 1.0
    elif hand_mode == 3: # 2に近いが、小さい数を1で埋めたもの 
        r = np.zeros(shape=(kind_tile, NUM_SAME_TILE), dtype=np.float32)
        for i in range(kind_tile):
            for j in range(hand[i]):
                r[i][j] = 1.0
    #print(self.hand)
    #print(r)
    return r    

