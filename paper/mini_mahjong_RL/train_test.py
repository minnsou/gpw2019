from baselines import deepq
from baselines import bench
from baselines import logger
import mahjong_env
import mahjong_utils
import tensorflow as tf
import numpy as np
import baselines.common.models
from baselines.common.models import register

@register("mydense")
def mydense_builder(num_layers=1, num_units=30, **dense_kwargs):
    def my_network(X):
        out = tf.cast(X, tf.float32)
        out = tf.contrib.layers.flatten(out)
        for num_l in range(num_layers):
            out = tf.contrib.layers.fully_connected(
                inputs=out,
                num_outputs=num_units,
                activation_fn=tf.nn.relu
            )
        return out
    return my_network

@register("myconv1d")
def myconv1d_builder(convs=[(32, 3, 1)], **conv1d_kwargs):
    def network_fn(X):
        out = tf.cast(X, tf.float32)
        for num_outputs, kernel_size, stride in convs:
            out = tf.contrib.layers.conv1d(
                inputs=out,
                num_outputs=num_outputs,
                kernel_size=kernel_size,
                stride=stride,
                padding='SAME',
                data_format='NWC',
                activation_fn=tf.nn.relu,  
            )
        return out
    return network_fn

@register("myconv2d")
def myconv2d_builder(convs=[(32, 3, 1)], **conv2d_kwargs):
    # convs = [(filter_number, filter_size, stride)]
    def network_fn(X):
        out = tf.cast(X, tf.float32)
        for num_outputs, kernel_size, stride in convs:
            out = tf.contrib.layers.convolution2d(
                inputs=out,
                num_outputs=num_outputs,
                kernel_size=kernel_size,
                stride=stride,
                padding='SAME',
                data_format='NHWC',
                activation_fn=tf.nn.relu,
            )
        return out
    return network_fn

def main():
    logger.configure()
    kind_tile = 9 # 用いる牌の種類数
    num_hand = 8 # 手牌の枚数
    NUM_SAME_TILE = 4 # 同一牌の枚数(4で固定)
    max_episode_step = 100
    hand_mode = 3 # NNに入れるときの手牌の形（1 or 2 or 3）
    simple_mode = False # Trueのときは役を使わない
    hiddens = [256]
    print_freq = 100
    network = 'mydense'
    num_layers = 2
    num_units = 512
    convs = [(64, 3, 1), (43, 3, 1)]
    total_timesteps = int(3 * 1e6)
    env = mahjong_env.MahjongEnv(process_idx=0, kind_tile=kind_tile, num_hand=num_hand, max_episode_step=max_episode_step, hand_mode=hand_mode, simple_mode=simple_mode, network=network, debug=False)
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
    '''.format(kind_tile, num_hand, max_episode_step, hand_mode, simple_mode, hiddens, print_freq, network, num_layers, num_units, convs, total_timesteps)
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
    model.save('save_models/save_model.pkl')

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
        discard_tile = model(one_hot)[0]
        #print(discard_tile)
        if discard_tile in ans_set:
            correct_num += 1
        else:
            wrong_num += 1
    print('total_num', total_num)
    print('correct num {}, wrong num {}'.format(correct_num, wrong_num))
    print('accuracy', correct_num / total_num)

    env.close()

if __name__ == '__main__':
    main()
