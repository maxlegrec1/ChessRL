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


# print the number of rows where index 0 is highest in values, the number of rows where index 1 is highest in values, and the number of rows where index 2 is highest in values
val_skewed = [(fen, val, move) for fen, val, move in zip(fens, values, moves) if val[0] > 0.7]
print(len(val_skewed))
# print(val_skewed[:10])
val_skewed = [(fen, val, move) for fen, val, move in zip(fens, values, moves) if val[2] > 0.7]
print(len(val_skewed))
print(val_skewed[:10])