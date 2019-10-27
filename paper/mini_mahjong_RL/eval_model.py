from baselines import deepq
from baselines import logger
import argparse
import mahjong_env
import mahjong_utils
import mahjong_networks
import tensorflow as tf
import numpy as np


def eval_optimal(act, simple_mode, num_hand, kind_tile, hand_mode, NUM_SAME_TILE):
    if simple_mode:
        path = 'optimal_list/simple_max_value_discard_list09_' + str(num_hand) + '.npy'
    else:
        path = 'optimal_list/yaku_max_value_discard_list09_' + str(num_hand) + '.npy'
    print('path =', path)
    optimal_npy = np.load(path)
    optimal_list = optimal_npy.tolist()
    total_num = len(optimal_list)
    
    correct_num = 0
    wrong_num = 0
    for discard_state, ans_set in optimal_list:
        #print(discard_state, ans_set)
        discard_hist = mahjong_utils.state_to_hist(discard_state, kind_tile)
        #print(discard_hist)
        one_hot = mahjong_utils.hist2onehot(hand_mode, num_hand, kind_tile, NUM_SAME_TILE, discard_hist)
        #print(one_hot)
        discard_tile = act(one_hot)[0]

        #print(discard_tile)
        if discard_tile in ans_set:
            correct_num += 1
        else:
            wrong_num += 1
    print('correct num {}, wrong num {} (total num {})'.format(correct_num, wrong_num, total_num))
    print('accuracy', correct_num / total_num)

    return


def main():
    logger.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hand", help="number of hand tile (defalut 8)", type=int, choices=[3, 5, 8, 11, 14], default=8)
    parser.add_argument("--hand_mode", help="choose hand_mode (defalut 3)", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--simple", help="simple mode", action="store_true", default=False)
    parser.add_argument("--hiddens", help="number of hidden units (default 125)", nargs='+', type=int, default=125)
    parser.add_argument("--network", help="choose a network (defalut 'mydense')", choices=['mydense', 'myconv1d', 'myconv2d'], default='mydense')
    parser.add_argument("--num_layers", help="number of layers (using only when 'mydense')", type=int, default=1)
    parser.add_argument("--num_units", help="number of units (using only when 'mydense')", type=int, default=302)
    parser.add_argument("--convs", help="number of filters (using when 'conv1d' or 'conv2d')", nargs='+', type=int, default=[64, 43])
    parser.add_argument("--total_timesteps", help="number of total timesteps (defalut 3000000)", type=int, default=int(3 * 1e6))
    parser.add_argument("--reward_scale", help="reward scale (defalut 100)", type=int,  default=100)
    args = parser.parse_args()
    
    kind_tile = 9 # 用いる牌の種類数
    num_hand = args.num_hand # 手牌の枚数
    NUM_SAME_TILE = 4 # 同一牌の枚数(4で固定)
    max_episode_step = 100
    hand_mode = args.hand_mode # NNに入れるときの手牌の形（1 or 2 or 3）
    simple_mode = args.simple # Trueのときは役を使わない
    if type(args.hiddens) == int:
        args.hiddens = [args.hiddens]
    hiddens = args.hiddens # リストで渡す
    print_freq = 100
    network = args.network
    num_layers = args.num_layers # mydenseで使う
    num_units = args.num_units # mydenseで使う
    if type(args.convs) == int:
        args.convs = [args.convs]
    conv_para = []
    for fil in args.convs:
        conv_para.append((fil, 3, 1))
    convs = conv_para # myconv1dとmyconv2dで使う
    total_timesteps = args.total_timesteps
    reward_scale = args.reward_scale # 報酬のスケール
    env = mahjong_env.MahjongEnv(process_idx=0, kind_tile=kind_tile, num_hand=num_hand, max_episode_step=max_episode_step, hand_mode=hand_mode, simple_mode=simple_mode, network=network, reward_scale=reward_scale, debug=False)
    print_str ='''
    kind_tile = {}
    num_hand = {}
    max_episode_step = {}
    hand_mode = {}
    simple_mode = {}
    hiddens = {}
    print_freq = {}
    network = {}
    num_layers = {}
    num_units = {} 
    convs = {}
    total_timesteps = {}
    reward_scale = {}
    '''.format(kind_tile, num_hand, max_episode_step, hand_mode, simple_mode, hiddens, print_freq, network, num_layers, num_units, convs, total_timesteps, reward_scale)
    print(print_str)


    if simple_mode:
        pkl_name = '{}{}_simple_{}.pkl'.format(network[2:], num_hand, hand_mode)
    else:
        pkl_name = '{}{}_yaku_{}.pkl'.format(network[2:], num_hand, hand_mode)
    print(pkl_name)
        
    act = deepq.learn(
        env,
        network,
        num_layers=num_layers,
        num_units=num_units,
        convs=convs,
        #batch_size=32,
        hiddens=hiddens,
        #dueling=True,
        #lr=1e-4,
        #total_timesteps=total_timesteps,
        #buffer_size=10000,
        #exploration_fraction=0.1,
        #exploration_final_eps=0.01,
        #train_freq=4,
        #learning_starts=10000,
        #target_network_update_freq=1000,
        gamma=0,
        print_freq=print_freq,
        total_timesteps=0,
        load_path="save_models/" + pkl_name
    )

    eval_optimal(act, simple_mode, num_hand, kind_tile, hand_mode, NUM_SAME_TILE)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        miss_count = 0
        epi_len = 0
        while not done:
            #env.render()
            old_obs = obs
            dis = act(obs[None])[0]
            obs, rew, done, _ = env.step(dis)
            #print(obs, rew, done)
            if rew < 0:
                print('wrong {}'.format(dis))
                for i in range(5):
                    print(i, act(old_obs[None])[0])
                miss_count += 1
            epi_len += 1
            episode_rew += rew
        print("Episode reward {} miss_count {} epi_len {} ratio {}".format(episode_rew, miss_count, epi_len, miss_count / epi_len))

    env.close()
        
if __name__ == '__main__':
    main()

    
