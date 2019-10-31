from baselines import deepq
from baselines import logger
import argparse
import mahjong_env
import mahjong_networks
import mahjong_utils
import eval_model
import tensorflow as tf


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
    model = deepq.learn(
        env,
        network,
        num_layers=num_layers,
        num_units=num_units,
        convs=convs,
        #batch_size=32,
        hiddens=hiddens,
        #dueling=True,
        #lr=1e-4,
        total_timesteps=total_timesteps,
        #buffer_size=10000,
        #exploration_fraction=0.1,
        #exploration_final_eps=0.01,
        #train_freq=4,
        #learning_starts=10000,
        #target_network_update_freq=1000,
        #gamma=0.90,
        print_freq=print_freq,
    )
    name_str = '{}{}_{}_{}.pkl'.format(network, num_hand, simple_mode, hand_mode)
    model.save('save_models/' + name_str)

    eval_model.eval_optimal(model, simple_mode, num_hand, kind_tile, hand_mode, NUM_SAME_TILE)

    env.close()

if __name__ == '__main__':
    main()
