import gym
import numpy as np
import mahjong_utils
import random
from gym.envs.registration import EnvSpec

class MahjongEnv(gym.Env):
    def __init__(self, process_idx=0, kind_tile=9, num_hand=5, max_episode_step=100, hand_mode=1, simple_mode=False, network='mydense', debug=True):
        self.process_idx = process_idx
        self.action_space = gym.spaces.Discrete(kind_tile)
        self.kind_tile = kind_tile
        self.num_hand = num_hand
        self.debug = debug
        self.spec = EnvSpec('Mahjong_{}_{}-v0'.format(kind_tile, num_hand))
        self.hand = [0] * kind_tile # 手牌はhistで管理
        self.max_episode_step = max_episode_step
        self.hand_mode = hand_mode # 1 or 2 or 3
        self.simple_mode = simple_mode
        if self.simple_mode:
            self.reward_range = (-1.0, 1.0)
        else:
            self.reward_range = (-1.0, 4.0)
        self.NUM_SAME_TILE = 4 # 同一牌の枚数は4で固定
        self.network = network
        if hand_mode == 1:
            ob_space_shape = [self.num_hand, self.kind_tile]
        elif hand_mode == 2:
            ob_space_shape = [self.kind_tile, self.NUM_SAME_TILE + 1]
        elif hand_mode == 3:
            ob_space_shape = [self.kind_tile, self.NUM_SAME_TILE]
        if self.network == 'myconv2d':
            ob_space_shape.append(1) # conv2dの時は次元を一つ増やす
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=tuple(ob_space_shape), dtype=np.float32)
            
    def draw_hand(self):
        while True:
            all_pi = np.array([i for i in range(self.kind_tile) for j in range(self.NUM_SAME_TILE)])
            all_pi = np.random.permutation(all_pi)[:self.num_hand]
            for i in range(self.kind_tile):
                self.hand[i] = np.count_nonzero(all_pi==i)
            if mahjong_utils.win_split_main(self.hand)[0]:
                if self.debug:
                    print('tenho', self.hand)
                continue
            break
        if self.debug:
            print('draw_hand', self.hand)
            
                
    def obs(self):
        if self.hand_mode == 1: # 牌の1つ1つをone-hotへ
            r = np.zeros(shape=(self.num_hand, self.kind_tile), dtype=np.float32)
            num = 0
            for idx, h in enumerate(self.hand):
                for i in range(h):
                    r[num][idx] = 1.0
                    num += 1
        elif self.hand_mode == 2: # hist形式の手牌をone-hotへ
            r = np.zeros(shape=(self.kind_tile, self.NUM_SAME_TILE + 1), dtype=np.float32)
            for i in range(self.kind_tile):
                r[i][self.hand[i]] = 1.0
        elif self.hand_mode == 3: # 2に近いが、小さい数を1で埋めたもの 
            r = np.zeros(shape=(self.kind_tile, self.NUM_SAME_TILE), dtype=np.float32)
            for i in range(self.kind_tile):
                for j in range(self.hand[i]):
                    r[i][j] = 1.0
        #print(self.hand)
        #print(r)
        return r    
            
    def reset(self):
        self.draw_hand()
        self.actions = []
        return self.obs()

    def step(self, act):
        self.actions.append(act)
        if len(self.actions) >= self.max_episode_step:
            if self.debug:
                print('finish episode(max episode step)')
            return self.obs(), 0.0, True, dict()
        if self.hand[act] == 0: # 切れない牌を切るときはマイナスの報酬
            if self.debug:
                print('no tile! hand = {}, action = {}'.format(self.hand, act))
            return self.obs(), -1.0, False, dict()
        self.hand[act] -= 1 # 牌を切る
        all_pi = np.array([i for i in range(self.kind_tile) for j in range(self.NUM_SAME_TILE - self.hand[i])])
        pi = all_pi[random.randrange(len(all_pi))]
        self.hand[pi] += 1 # 山から1枚ツモる
        if self.debug:
            print('hand', self.hand)
        is_agari, split_state = mahjong_utils.win_split_main(self.hand)
        if is_agari:
            if self.simple_mode:
                win_point = 1.0
            else:
                win_point = mahjong_utils.yaku_point(split_state)
            if self.debug:
                print('win! actions={}'.format(self.actions))
            return self.obs(), 1.0 * win_point, True, dict()
        return self.obs(), 0.0, False, dict()

