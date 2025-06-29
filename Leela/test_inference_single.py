from load_bt4 import create_model
from pgn_to_input_data import list_uci_to_input
from ustotheirs import ustotheirs
import tensorflow as tf
from gen_TC import policy_index
bt4 = create_model()


#moves = ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1","f8e7","f1e1","b7b5"]
moves = [
    "h2h4", "a7a5",
    "h4h5", "a5a4",
    "h5h6", "a4a3",
    "h6xg7", "a3xb2",
    "g7h8q", "b2a1q",
    "h8xg8", "a1xb1",  # White Q from h8 (promoted) captures g8. Black Q from a1 captures b1.
    "g8xh7", "b1xa2",  # White Q from g8 captures h7. Black Q from b1 captures a2.
    "h7d3", "a2e6",    # White original Q d1-d3. Black Q from a2 to e6. (Assuming original Q for White)
    "d3a3", "e6h6",    # White Q from d3 to a3. Black Q from e6 to h6.
    "a3xa8", "h6xh1",  # White Q from a3 captures a8. Black Q from h6 captures h1.
    "a8xb8", "h1xg1",  # White Q from a8 captures b8. Black Q from h1 captures g1.
    "b8xc8", "g1xg2",  # White Q from b8 captures c8. Black Q from g1 captures g2.
    "c8xb7", "g2c6",    # White Q from c8 captures b7. Black Q from g2 to c6.
    "b7xc7", "c6xc2",  # White Q from b7 captures c7. Black Q from c6 captures c2.
    "f1h3", "f8h6",    # White Bishop f1-h3. Black Bishop f8-h6. (c8 Bishop was captured)
    "h3xd7", "d8xd7",  # White Bishop from h3 captures d7. Black Q from c2 captures d7.
    "c7xd7", "e8xd7",  # White Q from c7 captures d7. Black K from e8 captures d7. (White Q on h7 remains)
    "f2f4", "h6xf4",    # Pawn f2-f4. Black B from h6 captures f4.
    "e2e3", "f4xe3",    # Pawn e2-e3. Black B from f4 captures e3.
    "d2xe3", "d7e6",    # Pawn d2 captures e3. Black K from d7 to e6.
    "d1xc2", "f7f5",    # White's remaining Q (that was on h7) captures c2. Pawn f7-f5.
    "c2xf5", "e6d6",    # White Q from c2 captures f5. Black K from e6 to d6.
    "c1a3", "d6c6",    # White Bishop c1-a3. Black K from d6 to c6. (f1 Bishop was captured on d7)
    "a3d6", "c6xd6",    # White Bishop from a3 to d6. Black K from c6 captures d6.
    "e3e4", "e7e6",    # Pawn e3-e4 (White pawn on e3 from 20.dxe3+). Pawn e7-e6.
    "f5xe6", "d6c7",    # White Q from f5 captures e6. Black K from d6 to c7.
    "e6g8", "c7b6",    # White Q from e6 to g8. Black K from c7 to kb6.
    "e4e5", "b6b5",    # Pawn e4-e5. Black K from kb6 to kb5.
    "e5e6", "b5b6",    # Pawn e5-e6. Black K from kb5 to kb6.
    "e6e7", "b6b7",    # Pawn e6-e7. Black K from kb6 to kb7.
    "e7e8r", "b7b6",   # Pawn e7 promotes to Rook on e8. Black K from kb7 to kb6.
    "e1f2", "b6b5",    # White King e1-f2. Black K from kb6 to kb5.
    "f2g3", "b5b4",    # White King f2-g3. Black K from kb5 to kb4.
    "g3g4", "b4c3",    # White King g3-g4. Black K from kb4 to kc3.
    "g4g5", "c3d3",    # White King g4-g5. Black K from kc3 to kd3.
    "g5g6", "d3d2",    # White King g5-g6. Black K from kd3 to kd2.
    "g8g7", "d2d1",    # White Q from g8 to g7. Black K from kd2 to kd1.
    "g6h7", "d1d2",    # White King g6-h7. Black K from kd1 to kd2.
    "e8g8", "d2e1",    # White Rook from e8 to g8. Black K from kd2 to ke1.
    "h7h8", "e1d1"     # White King h7-h8. Black King ke1-f2 (final Black move implied by PGN structure if it was 40...Kf2)
                        # The PGN ends on White's 40th move (Kh8). If there was a Black 40th move like Kf2, it would be "ke1f2".
                        # Given the PGN "40. Kh8", the list stops at "h7h8".
                        # If the PGN implies a full 40th move pair, then the last black move needs to be inferred or was truncated.
                        # Assuming the PGN provided is exactly as stated, ending with White's move.
]
#                                        starts here |
X,mask = list_uci_to_input(moves, 2000, '300')
Y_bt4 = bt4(ustotheirs(X[0]))
print(ustotheirs(X[0])[0,1,:,:])
softmax = tf.nn.softmax((mask+Y_bt4['policy']),axis=-1)
print(softmax.shape)
#best = tf.argmax((mask+Y_bt4['policy']),axis=-1)
#for id in best:
    #print(policy_index[id.numpy()])
topk = tf.argsort(softmax,axis=-1, direction='DESCENDING')
#print 10 best moves and their probabilities
for i in range(10):
    print(policy_index[topk[0][i].numpy()],softmax[0][topk[0][i].numpy()].numpy())
