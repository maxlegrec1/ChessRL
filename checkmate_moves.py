import chess
import chess.pgn
import csv
import random
import io # Needed to handle PGN strings if not reading directly from file path
from collections import deque

def extract_winning_side_near_mate_positions_to_file(pgn_path: str, csv_path: str,only_winning_side = True):
    """
    Reads a PGN file, identifies games ending in checkmate, and directly writes
    (fen, move) pairs played by the winning side from the final phase of
    those games to a CSV file without storing them in memory.

    Args:
        pgn_path (str): The file path to the input PGN file.
        csv_path (str): The file path where the output CSV should be saved.
    """
    games_processed = 0
    checkmate_games_found = 0
    positions_written = 0

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["FEN", "Move"]) # Write header

            with open(pgn_path, 'r', encoding='utf-8', errors='replace') as pgn_file:
                while True:
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break

                        games_processed += 1
                        board = game.board()
                        last_moves = deque(maxlen=15)

                        for node in game.mainline():
                            move = node.move
                            if move is None:
                                continue

                            fen_before = board.fen()
                            move_uci = move.uci()
                            current_color = board.turn
                            last_moves.append((fen_before, move_uci, current_color))

                            try:
                                board.push(move)
                            except Exception as e:
                                print(f"Warning: Illegal move {move_uci} encountered in game {games_processed} sequence. Skipping move. Error: {e}")
                                break

                        if board.is_checkmate():
                            checkmate_games_found += 1

                            result = game.headers.get("Result")
                            winning_color = None
                            if result == "1-0":
                                winning_color = chess.WHITE
                            elif result == "0-1":
                                winning_color = chess.BLACK

                            if winning_color is not None:
                                if only_winning_side:
                                    winning_side_relevant_moves = [(fen, move) for fen, move, color in last_moves if color == winning_color]
                                else:
                                    winning_side_relevant_moves = [(fen, move) for fen, move, color in last_moves]
                                num_to_sample = min(5, len(winning_side_relevant_moves))

                                if num_to_sample > 0:
                                    sampled_positions = random.sample(winning_side_relevant_moves, num_to_sample)
                                    writer.writerows(sampled_positions) # Write directly to CSV
                                    positions_written += len(sampled_positions)

                    except Exception as e:
                        print(f"Error processing game {games_processed + 1} in PGN: {e}")

    except FileNotFoundError:
        print(f"Error: PGN file not found at {pgn_path}")
        return
    except IOError as e:
        print(f"Error writing to CSV file {csv_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f"Processed {games_processed} games.")
    print(f"Found {checkmate_games_found} games ending in checkmate.")
    print(f"Successfully wrote {positions_written} near-mate positions from the winning side to {csv_path}")

# --- Example Usage ---
# Create a dummy PGN file for testing
dummy_pgn_content = """
[Event "Dummy Game 1"]
[Site "?"]
[Date "?"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "0-1"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7
11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 14. Ng3 g6 15. Bg5 h6 16. Bd2 Bg7 17. a4 c5 18. d5 c4 19. b4 cxb3
20. Bxb3 Nc5 21. Bc2 Qc7 22. Qe2 bxa4 23. Reb1 Reb8 24. c4 Bc8 25. Ba5 Qe7 26. Rxb8 Rxb8 27. Bxa4 Nfd7
28. Bc7 Rb4 29. Bxd7 Qxd7 30. Ba5 Rb3 31. Nd2 Rb2 32. Qe3 f5 33. exf5 gxf5 34. Nh5 f4 35. Qc3 Rb7
36. Nxg7 Qxg7 37. Kh2 Bf5 38. Re1 Kh7 39. Nf3 Ne4 40. Qc1 Nxf2 41. Nh4 Qg3+ 42. Kg1 Nh3+ 43. Kh1 Qxh4
44. gxh3 Qxh3+ 45. Kg1 Rg7+ 46. Kf2 Qg2# 0-1

[Event "Dummy Game 2 - Resignation"]
[Site "?"]
[Date "?"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "1-0"]
[Termination "Normal"]

1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. Qc2 O-O 5. a3 Bxc3+ 6. Qxc3 b6 7. Bg5 Bb7 8. e3 d6 9. Ne2 Nbd7 10. Rd1 Qe7
11. Ng3 h6 12. Bxf6 Nxf6 13. f3 c5 14. Be2 Rac8 15. O-O Rfd8 16. Rd2 cxd4 17. exd4 d5 18. b3 Qxa3 19. Ra1 Qe7
20. Rxa7 Ra8 21. Rda2 Rxa7 22. Rxa7 Ra8 23. Rxa8+ Bxa8 24. Qa1 Bb7 25. Qa7 Qb4 26. Qxb7 Qe1+ 27. Bf1 Qe3+ 28. Kh1 Qxd4
29. cxd5 Nxd5 30. Qb8+ Kh7 31. Qf8 Ne3 32. Qxf7 Nxf1 33. Nxf1 Qd1 34. Kg1 Qxb3 35. h4 b5 36. h5 Qd3 37. Qxe6 b4
38. Ne3 b3 39. Nf5 b2 40. Qg6+ Kh8 41. Qxg7# 1-0

[Event "Dummy Game 3 - Short Mate"]
[Site "?"]
[Date "?"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "1-0"]
[Termination "Normal"]

1. e4 e5 2. Qh5 Ke7 3. Qxe5# 1-0

[Event "Dummy Game 4 - Draw"]
[Site "?"]
[Date "?"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "1/2-1/2"]
[Termination "Normal"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e6 7. f3 Be7 8. Qd2 Qc7 9. O-O-O O-O 10. g4 Nc6
11. h4 b5 12. h5 Nd7 13. g5 Nde5 14. Be2 Na5 15. g6 Nac4 16. Bxc4 Nxc4 17. gxh7+ Kh8 18. Qd3 Nxe3 19. Qxe3 b4
20. Nce2 e5 21. Nf5 Bxf5 22. exf5 Rac8 23. Rd2 d5 24. Kb1 d4 25. Qd3 Bg5 26. Rdd1 Qc6 27. Rhg1 Rf7 28. Ng3 Rfc7 29. Ne4 Rfc7 30. Rg2 Bh6 31. Re2 a5 32. Rf1 a4 33. Rg1 b3 34. axb3 axb3 35. Qxb3 Qa6 36. Qa3 Qxe2 37. Qa5 Qxc2+ 38. Ka2 Qc4+
39. Kb1 Ra7 40. Qxa7 Qc2+ 41. Ka1 Qc1+ 42. Rxc1 Rxc1+ 43. Ka2 Ra1+ 44. Kxa1 d3 45. Qb8+ Kxh7 46. Qd8 d2 47. Nxd2 Bxd2 48. Qxd2
1/2-1/2
"""

# Define file paths for the example
dummy_pgn_file_path = "dummy_games_on_the_fly.pgn"
output_csv_file_path = "near_checkmate.csv"

# Write the dummy PGN content to a file
with open(dummy_pgn_file_path, "w", encoding='utf-8') as f:
    f.write(dummy_pgn_content)

pgn_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/casual_pgn/human.pgn"
# Run the modified function
extract_winning_side_near_mate_positions_to_file(pgn_path, output_csv_file_path,only_winning_side=False)

# You can then inspect 'winning_side_near_mate_positions_on_the_fly.csv'