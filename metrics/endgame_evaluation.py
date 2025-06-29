import chess
import torch
from model_bis import GPT, GPTConfig
from tqdm import tqdm

def calculate_endgame_score(model,file = "data/endgames2.csv",T = 1,num_positions = 1000,stockfish_path = "stockfish/stockfish-ubuntu-x86-64-avx2",depth = 15,is_thinking_model=True,limit_elo = False,elo = 1350):
    """
    Reads a file containing starting endgame positions that are all winning from the white side.
    Model will play against stockfish depth 15 and his winrate will be calculated.
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    if limit_elo:
        engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": elo  # Replace with desired Elo
        })
    bar = tqdm(total=num_positions)
    bar.set_description("Calculating endgame accuracy")
    points = 0
    positions = 0
    with open(file, "r") as f:
        lines = f.readlines()
    #skip first line
    lines = lines[1:]
    for i,line in enumerate(lines):
        positions += 1
        if positions >= num_positions:
            break
        board = chess.Board(line)
        while not board.is_game_over():
            #check side
            if board.turn == chess.WHITE:
                #model plays move
                fen = board.fen()
                if is_thinking_model:
                    move = model.get_move_from_fen(fen, T = T)
                else:
                    move = model.get_move_from_fen_no_thinking(fen, T = T)
                #print(move)
                try:
                    board.push_uci(move)
                except:
                    print(move,fen)
                    break
            else:
                fen = board.fen()
                result = engine.play(board, chess.engine.Limit(depth=depth))
                move = result.move.uci()
                #print(move)
                board.push_uci(move)

        bar.set_postfix_str(f"Accuracy: {points / (i+1):.2%}")
        bar.update(1)
        #print(line,board.result())
        try:
            if board.result() == "1-0":
                points += 1
        except:
            pass


    engine.quit()
    return points / (i+1)

if __name__ == "__main__":
    config = GPTConfig()

    #config.n_layer = 15
    #config.n_embd= 1024
    #config.n_head = 32
    config.vocab_size = 1929
    config.block_size = 256
    model = GPT(config).to("cuda")
    model.load_state_dict(torch.load("pretrain/follow_checkpoint_step_160000.pt"))
    #model.load_state_dict(torch.load("pretrain/checkpoint_step_10000.pt"))
    #model.load_state_dict(torch.load("pretrain/small_pretrain_bt4_hq_10000.pt"))
    #model.load_state_dict(torch.load("pretrain/pretrain_bt4_hq_10000.pt"))
    #model.load_state_dict(torch.load("pretrain/pretrain_bt4_40000.pt"))
    #model.load_state_dict(torch.load("pretrain/model.pt"))
    score = calculate_endgame_score(model,T = 0.1,is_thinking_model=False,limit_elo=False,num_positions=100)
    print(f"Score: {score * 100:.2f}%")