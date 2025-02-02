import pickle
import os
dir_path = "data_stockfish/"

for file in os.listdir(dir_path):
    path_file = os.path.join(dir_path,file)

    f = open(path_file,"rb")
    data = pickle.load(f)

    print(data['var'])