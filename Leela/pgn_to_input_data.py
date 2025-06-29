
import chess
import chess.pgn

import os
import sys
import numpy as np
import time

cwd = os.getcwd()
sys.path.append(cwd)

from gen_TC import get_x_from_board,swap_side,policy_index

def list_uci_to_input(list_of_moves,elo,TC):
    board = chess.Board()
    color = len(list_of_moves)

    X = []


    for i,m in enumerate(list_of_moves):
        real_move = board.parse_san(m)
        if i==len(list_of_moves)-1:
            #print("geto",board.turn == chess.WHITE)
            pass
        xs = get_x_from_board(elo,board,TC)
        board.push(real_move)
        if (color-i)%2 == 1:
            X.append(swap_side(xs[:,:,:12]))
        else:
            X.append(xs[:,:,:12])

    #add last position
    X.append(get_x_from_board(elo,board,TC))

    if board.turn == chess.BLACK:
        board = board.mirror()
    lm =   np.ones(1858,dtype=np.float32)*(-1000)
    for possible in board.legal_moves:
        possible_str = possible.uci()
        if possible_str[-1]!='n':
            lm[policy_index.index(possible_str)] = 0
        else:
             lm[policy_index.index(possible_str[:-1])] = 0


    X_final = np.concatenate([x[:,:,:12] for x in X[-8:-1]], axis = -1)
    X_final = np.concatenate([X_final,X[-1]], axis = -1)


    return [np.expand_dims(X_final[:,:,:-2],axis = 0),np.expand_dims(X_final[:,:,-2:],axis=0)],lm


if __name__=="__main__":
    game = ['Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8']
    print(len(game))
    t_0 = time.time()
    a,b = list_uci_to_input(game, 2000, '300')
    print(time.time()-t_0)
    print(a[0].shape,b.shape)