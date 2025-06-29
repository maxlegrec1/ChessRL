import torch
import chess
import chess.engine
import chess.pgn # Added for PGN handling
from datetime import datetime # Added for timestamping
from utils.parse import encode_fens
from utils.vocab import policy_index
from tqdm import tqdm
from math import log10

STOCKFISH_PATH = "stockfish/stockfish-ubuntu-x86-64-avx2"  # Adjust path if needed
STOCKFISH_ELO = 1400  # Set the desired ELO rating
N_GAMES = 200 # Reduced for faster testing, change back to 100 if needed

def get_stockfish_move(engine, board):
    result = engine.play(board, chess.engine.Limit(time=0.1))
    return result.move

def estimate_elo(win_rate, stockfish_elo):
    if win_rate == 0:
        return 0
    elif win_rate == 1:
        return 2500
    if win_rate == 0.5: # Avoid log10(1) which is 0, leading to division by zero if win_rate makes (1-win_rate)/win_rate = 1
        return stockfish_elo
    if win_rate > 0 and win_rate < 1: # ensure win_rate is not 0 or 1 to avoid math errors with log10
        return stockfish_elo - 400 * (log10((1 - win_rate) / win_rate))
    return "N/A"


def play_game(player1_name, player2_name, engine, model1, model2, device, temp=0.1, stockfish_elo=None, game_num=1, use_value=True):
    board = chess.Board()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = f"{player1_name} vs {player2_name}"
    pgn_game.headers["Site"] = "Local Machine"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_num)
    pgn_game.headers["White"] = player1_name
    pgn_game.headers["Black"] = player2_name

    if player1_name == "Stockfish" and stockfish_elo:
        pgn_game.headers["WhiteElo"] = str(stockfish_elo)
    if player2_name == "Stockfish" and stockfish_elo:
        pgn_game.headers["BlackElo"] = str(stockfish_elo)

    node = pgn_game # Current node in the PGN game tree

    while not board.is_game_over():
        current_player_name = ""
        model_to_use = None
        is_stockfish_turn = False

        if board.turn == chess.WHITE:
            current_player_name = player1_name
            model_to_use = model1
            if player1_name == "Stockfish":
                is_stockfish_turn = True
        else:
            current_player_name = player2_name
            model_to_use = model2
            if player2_name == "Stockfish":
                is_stockfish_turn = True

        move = None
        if is_stockfish_turn:
            move = get_stockfish_move(engine, board)
        else: # Model's turn
            if use_value:
                # Use the new value-based move selection
                move_uci = model_to_use.get_best_move_value(board.fen(), T=temp, device=device)
            else:
                # Use the existing policy-based move selection
                if model_to_use.is_thinking_model:
                    move_uci = model_to_use.get_move_from_fen(board.fen(), T=temp, device=device)
                else:
                    move_uci = model_to_use.get_move_from_fen_no_thinking(board.fen(), T=temp, device=device) # if not is_thinking

            if move_uci is None:
                print(f"Model ({current_player_name}) failed to produce a move for FEN: {board.fen()}")
                # Forfeit or error handling
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                pgn_game.headers["Result"] = result
                return result, pgn_game

            try:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    print(f"Invalid move by {current_player_name}: {move_uci} in {board.fen()}. Legal moves: {[m.uci() for m in board.legal_moves]}")
                    # Forfeit due to illegal move
                    result = "0-1" if board.turn == chess.WHITE else "1-0"
                    pgn_game.headers["Result"] = result
                    return result, pgn_game
            except ValueError:
                print(f"Invalid UCI string by {current_player_name}: {move_uci} in {board.fen()}")
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                pgn_game.headers["Result"] = result
                return result, pgn_game

            # print(f"{current_player_name} ({'White' if board.turn == chess.WHITE else 'Black'}) plays: {move.uci()}")
        
        if move is not None:
            board.push(move)
            node = node.add_variation(move) # Add move to PGN
        else: # Should not happen if logic is correct
            print("Error: Move was None but not handled.")
            result = "0-1" if board.turn == chess.WHITE else "1-0" # Generic error
            pgn_game.headers["Result"] = result
            return result, pgn_game

    result = board.result()
    pgn_game.headers["Result"] = result
    return result, pgn_game

def main_model_vs_stockfish(model = None,model1_name = "run",device = "cuda",num_games = N_GAMES,temp = 0.1,elo = STOCKFISH_ELO,use_value=False):
    if model is None:
        model_config = GPTConfig()
        model_config.vocab_size = 1929
        model_config.block_size = 256
        #model_config.n_layer = 15
        #model_config.n_embd = 1024
        #model_config.n_head = 32
        model = GPT(model_config).to(device)
        model.load_state_dict(torch.load("pretrain/follow_checkpoint_step_160000.pt", map_location=device))
        #model.load_state_dict(torch.load("pretrain/pretrain_bt4_40000.pt", map_location=device))

    wins, draws = 0, 0
    all_pgn_games = [] # List to store PGN game objects

    pgn_filename = datetime.now().strftime("games_model_vs_stockfish_%Y%m%d_%H%M%S.pgn")

    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            
            for i in tqdm(range(num_games), desc="Model vs Stockfish Games"):
                # Alternate who plays white
                if i % 2 == 0:
                    player1_model_name = "Model"
                    player2_stockfish_name = "Stockfish"
                    result, pgn_game = play_game(player1_model_name, player2_stockfish_name, engine, model, None, device, stockfish_elo=elo, game_num=i+1,temp=temp, use_value=use_value)
                    if result == "1-0": # Model (White) wins
                        wins += 1
                    elif result == "1/2-1/2":
                        draws += 1
                else:
                    player1_stockfish_name = "Stockfish"
                    player2_model_name = "Model"
                    result, pgn_game = play_game(player1_stockfish_name, player2_model_name, engine, None, model, device, stockfish_elo=elo, game_num=i+1,temp=temp, use_value=use_value)
                    if result == "0-1": # Model (Black) wins
                        wins +=1
                    elif result == "1/2-1/2":
                        draws += 1
                
                all_pgn_games.append(pgn_game)

    except FileNotFoundError:
        print(f"Error: Stockfish engine not found at {STOCKFISH_PATH}. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred during game play: {e}")
        # Optionally save any games played so far
        if all_pgn_games:
            print(f"Saving {len(all_pgn_games)} games played so far to {pgn_filename}")
            with open(pgn_filename, "w", encoding="utf-8") as f:
                for pgn_game in all_pgn_games:
                    exporter = chess.pgn.FileExporter(f)
                    pgn_game.accept(exporter)
                    f.write("\n\n") # Add some space between games


    print(f"Model wins: {wins}, Draws: {draws}, Losses: {num_games - wins - draws} out of {num_games} games.")
    if num_games > 0:
        win_rate = (wins + 0.5 * draws) / num_games
        estimated_elo = estimate_elo(win_rate, elo)
        print(f"Win rate against Stockfish ELO {elo}: {win_rate * 100:.2f}%")
        print(f"Estimated model ELO: {estimated_elo}")
    else:
        print("No games played.")

    # Save all games to a single PGN file
    with open(pgn_filename, "w", encoding="utf-8") as f:
        for pgn_game in all_pgn_games:
            exporter = chess.pgn.FileExporter(f)
            pgn_game.accept(exporter)
            f.write("\n\n") # Add some space between games for readability
    print(f"All games saved to {pgn_filename}")
    return win_rate, estimated_elo

def main_model_vs_model(model1 = None,model2 = None,model1_name = "run",model2_name = "run",device = "cuda",num_games = N_GAMES,temp = 0.1,use_value=False):
    if model1 is None:
        model_config = GPTConfig()
        model_config.vocab_size = 1929
        model_config.block_size = 256
        model_config.n_layer = 15
        model_config.n_embd = 1024
        model_config.n_head = 32
        model1_name = "200M_A_40k" # Descriptive name for PGN
        model1_path = "pretrain/pretrain_bt4_40000.pt"
        model1 = GPT(model_config).to(device)
        model1.load_state_dict(torch.load(model1_path, map_location=device))

    if model2 is None:
        model2_path = "pretrain/follow_checkpoint_step_160000.pt" # Change model if needed
        model2_name = "40M_B_pretrain" # Descriptive name for PGN
        model_config = GPTConfig()
        model_config.vocab_size = 1929
        model_config.block_size = 256
        model2 = GPT(model_config).to(device)
        model2.load_state_dict(torch.load(model2_path, map_location=device))


    model1_wins, draws = 0, 0
    all_pgn_games = []

    pgn_filename = datetime.now().strftime("games_model_vs_model_%Y%m%d_%H%M%S.pgn")
    
    # Dummy engine for play_game compatibility, not used if no stockfish player
    # If your `play_game` absolutely needs an engine even for model vs model, ensure it's handled.
    # For this setup, we can pass None or a dummy engine if the function expects it.
    # The provided `play_game` only uses engine if a player is "stockfish".
    # We pass None for engine as it's not used in model vs model.
    # However, the original code snippet used `with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:`
    # This implies it might be needed or was just copied over.
    # For safety, let's keep it but it won't be used by `get_stockfish_move`.
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine: # Stockfish engine might not be strictly needed here
            for i in tqdm(range(num_games), desc="Model vs Model Games"):
                current_pgn_game = None
                result = None
                if i % 2 == 0: # Model1 as White, Model2 as Black
                    # print(f"Game {i+1}: {model1_name} (White) vs {model2_name} (Black)")
                    result, current_pgn_game = play_game(model1_name, model2_name, engine, model1, model2, device, game_num=i+1, temp=temp, use_value=use_value)
                    if result == "1-0":
                        model1_wins += 1
                    elif result == "1/2-1/2":
                        draws += 1
                else: # Model2 as White, Model1 as Black
                    # print(f"Game {i+1}: {model2_name} (White) vs {model1_name} (Black)")
                    result, current_pgn_game = play_game(model2_name, model1_name, engine, model2, model1, device, game_num=i+1, temp=temp, use_value=use_value)
                    if result == "0-1": # Model1 (Black) wins
                        model1_wins += 1
                    elif result == "1/2-1/2":
                        draws += 1
                
                if current_pgn_game:
                    all_pgn_games.append(current_pgn_game)
    
    except FileNotFoundError:
        print(f"Warning: Stockfish engine not found at {STOCKFISH_PATH}. This is okay for model vs model if Stockfish isn't playing.")
        # If you want to proceed without Stockfish for model vs model, remove the `return`
        # For now, assuming it's okay if it's not used.
    except Exception as e:
        print(f"An error occurred during game play: {e}")
        if all_pgn_games:
            print(f"Saving {len(all_pgn_games)} games played so far to {pgn_filename}")
            with open(pgn_filename, "w", encoding="utf-8") as f:
                for pgn_game_obj in all_pgn_games:
                    exporter = chess.pgn.FileExporter(f)
                    pgn_game_obj.accept(exporter)
                    f.write("\n\n")
        return


    if N_GAMES > 0:
        win_rate_model1 = (model1_wins + 0.5 * draws) / num_games
        print(f"Results for {model1_name} vs {model2_name}:")
        print(f"{model1_name} Wins: {model1_wins}")
        print(f"Draws: {draws}")
        print(f"{model2_name} Wins: {num_games - model1_wins - draws}")
        print(f"Win rate ({model1_name} vs {model2_name}): {win_rate_model1 * 100:.2f}%")
    else:
        print("No games played.")

    # Save all games to a single PGN file
    with open(pgn_filename, "w", encoding="utf-8") as f:
        for pgn_game_obj in all_pgn_games:
            exporter = chess.pgn.FileExporter(f)
            pgn_game_obj.accept(exporter)
            f.write("\n\n") # Add some space between games for readability
    print(f"All games saved to {pgn_filename}")

    del model2
    return win_rate_model1


if __name__ == "__main__":
    # Ensure your model files (e.g., GPT, GPTConfig) and utils are in the PYTHONPATH
    # or in the same directory.
    # Example:
    # from .model_bis import GPT, GPTConfig # if in a package
    # from .utils.parse import encode_fens  # if in a package
    # from .utils.vocab import policy_index # if in a package
    
    # Assuming model_bis.py, utils/parse.py, utils/vocab.py are accessible.
    # You might need to create empty __init__.py files in 'utils' if it's a package.

    # Uncomment one of the following lines to run the desired test:
    from models.model import BT4
    from utils.parse import fen_to_tensor

    
    model1 = BT4().to("cuda")
    model1.load_state_dict(torch.load(f"pretrain/value_only_new_44000.pt",map_location="cuda"))

    fens = ["1nb1k2r/4qp2/p2p2p1/p1pPpPn1/1rP1P1Pp/R1N2N1P/4B1K1/5R2 w - - 0 26"]
    print(model1.get_best_move_value(fens[0],T=0,device="cuda"))
    #fens = ["8/1kp5/3p4/4n3/Q7/1p1r4/4K3/8 w - - 2 56"]
    position_tensors = []
    for fen in fens:
        x = torch.from_numpy(fen_to_tensor(fen)).to("cuda").to(torch.float32)
        position_tensors.append(x)
    
    # Stack to create batch: [batch_size, 8, 8, 19]
    batch_x = torch.stack(position_tensors, dim=0)
    # Reshape to [batch_size, 1, 8, 8, 19] for the model
    batch_x = batch_x.unsqueeze(1)
    
    # Forward pass through the model to get values
    with torch.no_grad():
        b, seq_len, _, _, emb = batch_x.size()
        x_processed = batch_x.view(b * seq_len, 64, emb)
        x_processed = model1.linear1(x_processed)
        x_processed = torch.nn.GELU()(x_processed)
        x_processed = model1.layernorm1(x_processed)
        x_processed = model1.ma_gating(x_processed)
        
        pos_enc = model1.positional(x_processed)
        for i in range(model1.num_layers):
            x_processed = model1.layers[i](x_processed, pos_enc)
        
        value_logits = model1.value_head(x_processed)
        value_logits = value_logits.view(b, seq_len, 3)
        value_logits = torch.softmax(value_logits, dim=-1)

    print(value_logits.shape,value_logits)
    exit()
    score = main_model_vs_stockfish(model=model1,model1_name="run",device="cuda",num_games=40,temp=0,use_value=True)
    print(score)
    #main_model_vs_model(model1=model1,model2=model2,model1_name=model1_name,model2_name=model2_name,device="cuda",num_games=200,temp=0.5)
