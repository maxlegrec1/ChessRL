
import chess
import chess.pgn

import os
import sys
import numpy as np
import time

cwd = os.getcwd()
sys.path.append(cwd)

from gen_TC import get_x_from_board,swap_side,policy_index

def list_uci_to_input(pre_pos,post_pos,elo,TC):
    list_of_moves = pre_pos + post_pos
    start_index = len(pre_pos)+1
    assert start_index >= 8
    board = chess.Board()
    color = len(list_of_moves)
    X_finals = []
    X = []
    fens = []
    masks = []
    for i,m in enumerate(list_of_moves): 
        real_move = board.parse_san(m)
        if i==len(list_of_moves)-1:
            #print("geto",board.turn == chess.WHITE)
            pass
        xs = get_x_from_board(elo,board,TC)
        if board.turn == chess.BLACK:
            mirrored_board = board.mirror()
        else:
            mirrored_board = board
        lm = np.ones(1858,dtype=np.float32)*(-1000)
        for possible in mirrored_board.legal_moves:
            possible_str = possible.uci()
            if possible_str[-1]!='n':
                lm[policy_index.index(possible_str)] = 0
            else:
                lm[policy_index.index(possible_str[:-1])] = 0
        masks.append(lm)
        fens.append(board.fen())    
        board.push(real_move)
        X.append(xs)

    #add last position
    X.append(get_x_from_board(elo,board,TC))
    fens.append(board.fen())
    if board.turn == chess.BLACK:
        board = board.mirror()
    lm =   np.ones(1858,dtype=np.float32)*(-1000)
    for possible in board.legal_moves:
        possible_str = possible.uci()
        if possible_str[-1]!='n':
            lm[policy_index.index(possible_str)] = 0
        else:
             lm[policy_index.index(possible_str[:-1])] = 0
    masks.append(lm)
    #print(len(X),len(masks))
    for i in range(start_index-8,len(X)-8):
        X_final = []
        for j in range(i,i+7):
            #print(j)
            if (i-j)%2 == 0:
                X_final.append(swap_side(X[j][:,:,:12]))
            else:
                X_final.append(X[j][:,:,:12])
        #print(i+7)
        X_final = np.concatenate(X_final, axis = -1)
        X_final = np.concatenate([X_final,X[i+7]], axis = -1)
        X_finals.append(X_final)

    X_finals = np.array(X_finals)
    masks = np.array(masks[start_index-1:-1])

    return [X_finals,None],masks,fens[start_index-1:-1]


if __name__=="__main__":
    game = ['Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8','Nf3','Nf6','Ng1','Ng8']
    print(len(game))
    t_0 = time.time()
    a,b = list_uci_to_input(game, 2000, '300')
    print(time.time()-t_0)
    print(a.shape,b.shape)