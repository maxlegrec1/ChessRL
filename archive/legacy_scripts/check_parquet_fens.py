import pyarrow.parquet as pq

table = pq.read_table("src/self_play/selfplay_buffer.parquet")
fens = table.column("fen").to_pylist()
#check for duplicates, print number of fens that are there more than once
set_fens = set(fens)
print(len(set_fens))
print(len(fens))

# print(set_fens)
#print table keys
# print(table.schema.names)
values = table.column("value").to_pylist()
moves = table.column("policy").to_pylist()

knight_promitions = [move for move in moves if "n" in move]
print(f"Number of knight promotions: {len(knight_promitions)}")


# print the number of rows where index 0 is highest in values, the number of rows where index 1 is highest in values, and the number of rows where index 2 is highest in values
val_skewed = [(fen, val, move) for fen, val, move in zip(fens, values, moves) if val[0] > 0.7]
print(len(val_skewed))
# print(val_skewed[:10])
val_skewed = [(fen, val, move) for fen, val, move in zip(fens, values, moves) if val[2] > 0.7]
print(len(val_skewed))
# print(val_skewed[:10])


argmax0 = [(fen, val, move) for fen, val, move in zip(fens, values, moves) if val[0] > val[1] and val[0] > val[2] and val[0]]

argmax1 = [(fen, val, move) for fen, val, move in zip(fens, values, moves) if val[1] > val[0] and val[1] > val[2] and val[1]]

argmax2 = [(fen, val, move) for fen, val, move in zip(fens, values, moves) if val[2] > val[0] and val[2] > val[1] and val[2]]


print(len(argmax0), len(argmax1), len(argmax2))

# print % of fens that are white to move in argmax0
print(len([fen for fen, val, move in argmax0 if fen.split(" ")[1] == "w"]) / len(argmax0))

# print % of fens that are black to move in argmax2
print(len([fen for fen, val, move in argmax2 if fen.split(" ")[1] == "b"]) / len(argmax2))

print(argmax0[-10:])