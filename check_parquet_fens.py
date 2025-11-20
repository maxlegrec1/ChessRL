import pyarrow.parquet as pq

table = pq.read_table("src/self_play/selfplay_buffer.parquet")
fens = table.column("fen").to_pylist()
#check for duplicates, print number of fens that are there more than once
set_fens = set(fens)
print(len(set_fens))
print(len(fens))

# print(set_fens)