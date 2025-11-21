import pyarrow.parquet as pq

table = pq.read_table("src/self_play/selfplay_buffer.parquet")
fens = table.column("fen").to_pylist()
#check for duplicates, print number of fens that are there more than once
set_fens = set(fens)
print(len(set_fens))
print(len(fens))

# print(set_fens)

values = table.column("value").to_pylist()
# print(values)

# print the number of rows where index 0 is highest in values, the number of rows where index 1 is highest in values, and the number of rows where index 2 is highest in values
print(len([val for val in values if val[0] > val[1] and val[0] > val[2]]))
print(len([val for val in values if val[1] > val[0] and val[1] > val[2]]))
print(len([val for val in values if val[2] > val[0] and val[2] > val[1]]))