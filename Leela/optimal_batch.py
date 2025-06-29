

from load_bt4 import create_model
import numpy as np
import time
import matplotlib.pyplot as plt
bt4 = create_model()
mean_time_per_pos = []
max_batch_size = 1024
min_batch_size = 256

for batch_size in range(min_batch_size, max_batch_size+1):
    X = np.random.rand(batch_size, 112, 8, 8).astype(np.float32)
    for i in range(10):
        t0 = time.time()
        Y_bt4 = bt4(X)
        t1 = time.time()
        print("Batch size:", batch_size, "Time:", t1-t0)
        if i == 0:
            times = [t1-t0]
        else:
            times.append(t1-t0)
    mean_time_per_pos.append(np.mean(times)/batch_size)
plt.plot(range(min_batch_size, max_batch_size+1), mean_time_per_pos)
plt.xlabel("Batch size")
plt.ylabel("Mean time per position (s)")
plt.title("Mean time per position vs batch size")
plt.savefig("mean_time_per_pos.png")