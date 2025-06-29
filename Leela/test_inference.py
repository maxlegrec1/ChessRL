from load_bt4 import create_model
from pgn_to_input_data_decoder import list_uci_to_input
from ustotheirs import ustotheirs
import tensorflow as tf
from gen_TC import policy_index
from compare_policy import map_leela_to_rl
from tqdm import tqdm
import time
bt4 = create_model()



for _ in tqdm(range(100000000)):
    pre_pos = ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4"]
    post_pos = ["g8f6","e1g1","f8e7","f1e1","b7b5","a4b3","e8g8"]
    t0 = time.time()
    X,mask,fens = list_uci_to_input(pre_pos,post_pos,2000, '300')
    t1 = time.time()
    X = ustotheirs(X[0])
    t2 = time.time()
    Y_bt4 = bt4(X)
    t3 = time.time()
    policies = tf.nn.softmax((mask+Y_bt4['policy']),axis=-1)
    t4 = time.time()
    policies_rl = map_leela_to_rl(policies)
    t5 = time.time()
    print(policies_rl.shape)
    #print all time differences:
    print("list_uci_to_input:",t1-t0)
    print("ustotheirs:",t2-t1)
    print("bt4:",t3-t2)
    print("softmax:",t4-t3)
    print("map_leela_to_rl:",t5-t4)

    best = tf.argmax((mask+policies['policy']),axis=-1)
    for id,fen in zip(best,fens):
        print(fen,policy_index[id.numpy()])

