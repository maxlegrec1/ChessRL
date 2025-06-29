import client_final_final
import pickle
import numpy as np # If expecting numpy arrays

# The connection is already established upon import (if successful)

# Send some moves
pre_moves = ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4"]
post_moves = ["g8f6","e1g1","f8e7","f1e1","b7b5","a4b3","e8g8"]
payload = client_final_final.send_moves(pre_moves, post_moves)
if payload is not None:
    print("Payload received (first 100 bytes):", payload[:100])
    # Deserialize the response
    try:
        prediction = pickle.loads(payload)
        print(prediction.shape)
    except pickle.UnpicklingError as e:
        print(f"Error unpickling response: {e}")
    except Exception as e:
         print(f"Unexpected error unpickling response: {e}")


