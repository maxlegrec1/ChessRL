import socket
import pickle
import tensorflow as tf
import struct # Import struct
from load_bt4 import create_model # Ensure accessible
from pgn_to_input_data_decoder import list_uci_to_input # Ensure accessible
from ustotheirs import ustotheirs # Ensure accessible
from gen_TC import policy_index # Ensure accessible
from compare_policy import map_leela_to_rl
import numpy as np

# --- Configuration ---
HOST = '127.0.0.1'
PORT = 65432
BUFFER_SIZE = 4096
HEADER_SIZE = struct.calcsize('>Q') # Should be 8 bytes

import os, psutil
import tracemalloc
tracemalloc.start()
import gc
proc = psutil.Process(os.getpid())

def log_mem(stage):
    mem_mb = proc.memory_info().rss / (1024*1024)
    print(f"[MEMORY] {stage}: {mem_mb:.1f} MiB")



# --- Model Loading --- (Keep as is)
print("Chargement du modèle TensorFlow...")
try:
    bt4 = create_model()
    print("Modèle chargé.")
    log_mem("after model load")
except Exception as e:
    print(f"Erreur lors du chargement ou du test du modèle: {e}")
    exit()


def batched_inference(model, X, batch_size):
    outputs = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        output_batch = model(X_batch, training=False)['policy']  # Use `training=False` to disable dropout, etc.
        outputs.append(output_batch)
    return tf.concat(outputs, axis=0)

# --- Prediction Function --- (Keep as is)
def predict_move(policies_array):
    """Prédit le meilleur coup suivant à partir d'une liste de coups UCI."""
    #print(f"  Traitement de la liste de coups: {policies_array}")
    try:
        big_X,big_mask= [],[]
        for policies in policies_array:
            pre_pos = policies[0]
            post_pos = policies[1]
            #print(f"  Traitement de la liste de coups: {pre_pos + post_pos}")
            # Convertir la liste de coups en entrée pour le modèle
            # Adapter les valeurs 2000 et '300' si nécessaire
            X, mask,fens = list_uci_to_input(pre_pos,post_pos, 2000, '300')
            big_X.append(X[0])
            big_mask.append(mask)

        big_X = tf.concat(big_X,axis=0)
        big_mask = tf.concat(big_mask,axis=0)
        Y_bt4 = batched_inference(bt4,tf.cast(ustotheirs(big_X),dtype = tf.float16), 2048) # Use batch size of 32
        #Y_bt4 = bt4(ustotheirs(big_X))
        policies = tf.nn.softmax((big_mask + Y_bt4), axis=-1)
        k_list = [len(el[1]) for el in policies_array]
        split_indices = np.cumsum(k_list)[:-1]
        #print(len(policies),split_indices)
        policies = tf.split(policies, k_list, axis=0)
        print("mapping")
        policies = [map_leela_to_rl(policies[i],first_is_white=len(policies_array[i][0])%2==0).astype(np.float16) for i in range(len(policies))]
        return policies
    except Exception as e:
        print(f"  Erreur pendant la prédiction: {e}")
        import traceback
        traceback.print_exc() # Print full error for debugging
        return None

# --- Helper function to receive exactly n bytes ---

def recv_all(sock, n):
    """Helper function to receive exactly n bytes from sock"""
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None # Connection closed prematurely
        data += packet
    return data

'''
def recv_all(sock, n):
    # 1️⃣ allocate once
    buf = bytearray(n)
    view = memoryview(buf)
    total = 0

    # 2️⃣ fill it in place
    while total < n:
        got = sock.recv_into(view[total:], n - total)
        if got == 0:
            return None
        total += got

    # 3️⃣ (optional) convert to immutable bytes for pickle.loads
    return bytes(buf)
'''

# --- Mise en place du serveur ---
print(f"Démarrage du serveur sur {HOST}:{PORT}")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Optional: Allows faster restart
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print("Serveur en attente de connexions...")

    while True: # Main loop to accept new client connections
        conn = None # Ensure conn is defined for finally block
        try:
            conn, addr = server_socket.accept()
            print(f"\nConnecté par {addr}")
            with conn: # Use context manager for the accepted connection socket
                 while True: # Loop to handle multiple requests from THIS client
                    # --- Receive header ---
                    print(f"Server: Waiting for header ({HEADER_SIZE} bytes) from {addr}...")
                    header_data = recv_all(conn, HEADER_SIZE)
                    if header_data is None:
                        print(f"Server: Client {addr} disconnected before sending header.")
                        break # Exit loop for this client

                    payload_length = struct.unpack('>Q', header_data)[0]
                    print(f"Server: Received header. Expecting payload of {payload_length} bytes from {addr}.")

                    # --- Receive payload ---
                    payload_data = recv_all(conn, payload_length)
                    if payload_data is None:
                        print(f"Server: Client {addr} disconnected before sending full payload (expected {payload_length} bytes).")
                        break # Exit loop for this client

                    print(f"Server: Received complete payload ({len(payload_data)} bytes) from {addr}.")

                    # --- Désérialisation ---
                    try:
                        policies = pickle.loads(payload_data)
                        print(f"Server: Data from {addr} deserialized successfully.")

                    except pickle.UnpicklingError as e:
                        print(f"Server: Erreur de désérialisation Pickle from {addr}: {e}")
                        # Maybe send error back?
                        continue # Wait for next message
                    except Exception as e:
                        print(f"Server: Erreur inattendue lors de la désérialisation from {addr}: {e}")
                        # Maybe send error back?
                        continue # Wait for next message

                    # --- Traitement et Prédiction ---
                    policies = predict_move(policies)
                    log_mem("after predict_move")
                    # --- Sérialisation et Envoi de la réponse (with length prefix) ---
                    if policies is not None:
                        print(f"Server: Type of response: {type(policies)}")
                        if isinstance(policies, list):
                            print(f"Server: Length of list: {len(policies)}")
                            print(f"Server: First element type: {type(policies[0])}")
                            try:
                                print(f"Server: Shape of first element: {policies[0].shape}")
                            except AttributeError:
                                pass

                        response_payload = pickle.dumps(policies)
                        response_length = len(response_payload)
                        response_header = struct.pack('>Q', response_length)

                        print(f"Server: Sending response header ({len(response_header)} bytes, length={response_length}) to {addr}")
                        conn.sendall(response_header)
                        print(f"Server: Sending response payload ({response_length} bytes) to {addr}")
                        conn.sendall(response_payload)
                        del header_data, payload_data, policies, response_payload

                        print(f"Server: Réponse envoyée à {addr}.")
                    else:
                        print(f"Server: Prediction failed for {addr}. Sending empty response.")
                        # Send a header indicating 0 length payload
                        conn.sendall(struct.pack('>Q', 0))

                    print(f"Server: Ready for next request from {addr}...")

        except socket.error as e:
            print(f"Server: Socket error communicating with {addr if conn else 'Unknown'}: {e}")
            # The 'with conn:' block will handle closing the connection if it exists
        except ConnectionResetError:
             print(f"Server: Client {addr} forcibly closed the connection.")
        except KeyboardInterrupt:
            print("\nArrêt du serveur demandé.")
            break # Exit the main server loop
        except Exception as e:
            print(f"Server: Erreur inattendue dans la boucle d'acceptation/traitement: {e}")
            import traceback
            traceback.print_exc() # Log the full error
        finally:
            # The 'with conn:' takes care of closing the connection for the current client
            # If error happened before 'with conn', conn might be None
            print(f"Server: Finished with client {addr if conn else 'Unknown'} or loop iteration.")


print("Serveur arrêté.")