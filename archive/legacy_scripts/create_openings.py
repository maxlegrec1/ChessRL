import chess
import chess.engine
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def generate_opening():
    STOCKFISH_PATH = "/mnt/2tb/ChessRL/stockfish/stockfish-ubuntu-x86-64-avx2"  # Adjust path if needed
    STOCKFISH_ELO = int(random.uniform(1500, 2900))
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})
    moves = []

    def get_stockfish_move(engine_obj, bullet_board):
        # Use python-chess board for Stockfish, then convert back to bulletchess.Move
        ch_board = chess.Board(bullet_board.fen())
        result = engine_obj.play(ch_board, chess.engine.Limit(time=0.1))
        return result.move.uci()


    board = chess.Board()

    num_moves = random.randint(10, 20)

    for i in range(num_moves):
        move = get_stockfish_move(engine, board)
        board.push_uci(move)
        moves.append(move)
        # print(board.fen())

    engine.quit()
    return board.fen(), moves


if __name__ == "__main__":
    import pyarrow.parquet as pq
    import pyarrow as pa
    from tqdm import tqdm
    
    parquet_filename = "openings.parquet"
    parquet_schema = pa.schema(
        [
            ("fen", pa.string()),
            ("moves", pa.list_(pa.string())),
        ]
    )
    
    # Configuration
    NUM_OPENINGS = 10000
    NUM_THREADS = 8  # Adjust based on your CPU cores
    BATCH_SIZE = 1000  # Write to parquet in batches
    
    parquet_writer = pq.ParquetWriter(parquet_filename, parquet_schema)
    write_lock = Lock()
    results_buffer = []
    
    def generate_and_buffer(_):
        """Generate opening and return result for batching"""
        try:
            return generate_opening()
        except Exception as e:
            print(f"Error generating opening: {e}")
            return None
    
    # Use ThreadPoolExecutor to generate openings in parallel
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all tasks
        futures = {executor.submit(generate_and_buffer, i): i for i in range(NUM_OPENINGS)}
        
        # Process completed tasks with progress bar
        with tqdm(total=NUM_OPENINGS, desc="Generating openings") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    fen, moves = result
                    with write_lock:
                        results_buffer.append({"fen": fen, "moves": moves})
                        
                        # Write batch when buffer is full
                        if len(results_buffer) >= BATCH_SIZE:
                            batch_data = {
                                "fen": [r["fen"] for r in results_buffer],
                                "moves": [r["moves"] for r in results_buffer],
                            }
                            parquet_writer.write_table(
                                pa.Table.from_pydict(batch_data, schema=parquet_schema)
                            )
                            results_buffer.clear()
                
                pbar.update(1)
    
    # Write remaining results
    if results_buffer:
        batch_data = {
            "fen": [r["fen"] for r in results_buffer],
            "moves": [r["moves"] for r in results_buffer],
        }
        parquet_writer.write_table(
            pa.Table.from_pydict(batch_data, schema=parquet_schema)
        )
    
    parquet_writer.close()