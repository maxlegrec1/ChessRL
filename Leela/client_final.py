import socket
import pickle
import atexit # Import the atexit module for cleanup
import numpy as np # Keep numpy if the server expects/sends numpy arrays

# --- Configuration ---
SERVER_HOST = '127.0.0.1'  # Server hostname or IP
SERVER_PORT = 65432        # Server port
BUFFER_SIZE = 4096

# --- Global variable to hold the socket connection ---
_client_socket = None

# --- Function to close the connection ---
def _close_connection():
    """Closes the global socket connection if it's open."""
    global _client_socket
    if _client_socket:
        print("Closing connection to server...")
        try:
            # Optionally send a 'disconnect' message if your server expects one
            # _client_socket.sendall(pickle.dumps("disconnect"))
            _client_socket.shutdown(socket.SHUT_RDWR) # Signal closure
            _client_socket.close()
            _client_socket = None
            print("Connection closed.")
        except socket.error as e:
            print(f"Error closing socket: {e}")
        except Exception as e:
            print(f"Unexpected error during socket close: {e}")

# --- Register the cleanup function to run on script exit ---
atexit.register(_close_connection)

# --- Connect to the server when the module is imported ---
try:
    print(f"Attempting to connect to server {SERVER_HOST}:{SERVER_PORT}...")
    _client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _client_socket.connect((SERVER_HOST, SERVER_PORT))
    print("Successfully connected to the server.")
    # Keep the connection open until the program exits

except socket.error as e:
    print(f"FATAL: Failed to connect to server during import: {e}")
    _client_socket = None # Ensure socket is None if connection failed
except ConnectionRefusedError:
    print(f"FATAL: Connection refused. Is the server ({SERVER_HOST}:{SERVER_PORT}) running and accessible?")
    _client_socket = None
except Exception as e:
    print(f"FATAL: An unexpected error occurred during initial connection: {e}")
    _client_socket = None

# --- Function to send moves and receive response ---
def send_moves(pre_pos, post_pos):
    """
    Sends pre_pos and post_pos lists to the connected server and returns the raw response data.

    Args:
        pre_pos (list): A list of previous position strings.
        post_pos (list): A list of post position strings.

    Returns:
        bytes: The raw data received from the server, or None if an error occurred.
    """
    global _client_socket

    if not _client_socket:
        print("Error: No active connection to the server.")
        return None

    try:
        # --- Serialize the data for this specific request ---
        data_to_send_pickle = pickle.dumps((pre_pos, post_pos))
        print(f"Sending data ({len(data_to_send_pickle)} bytes): {(pre_pos, post_pos)}")

        # --- Send data over the existing connection ---
        _client_socket.sendall(data_to_send_pickle)
        print("Data sent. Waiting for response...")

        # --- Signal end of sending for this request ---
        # This helps the server know the client is done sending *this* message.
        # Important if the server reads until SHUT_WR or EOF on its side for each request.
        _client_socket.shutdown(socket.SHUT_WR)

        # --- Receive the response ---
        received_data = b""
        while True:
            # Add a timeout to prevent hanging indefinitely if server misbehaves
            # _client_socket.settimeout(10.0) # 10 second timeout
            try:
                chunk = _client_socket.recv(BUFFER_SIZE)
                if not chunk:
                    # Server closed the *read* side after sending its response for this request.
                    # We need to re-enable reading for the *next* request.
                    # This is tricky - the server needs to be designed for persistent connections.
                    # Often, the server *won't* close the connection fully.
                    # If the server *does* close the connection after each reply,
                    # this single-connection approach won't work without reconnecting.
                    # Assuming the server keeps the connection open after replying:
                    break # Exit the loop once the current response is fully received.
                received_data += chunk
            except socket.timeout:
                print("Error: Socket timeout while waiting for server response.")
                return None # Indicate timeout error
            # except BlockingIOError:
            #     # Handle non-blocking sockets if you were using them
            #     pass
            finally:
                 # Remove timeout for subsequent operations if set
                 # _client_socket.settimeout(None)
                 pass # No timeout used in this version


        if not received_data:
            print("Warning: No response data received from the server for this request.")
            # Need to re-open the socket for writing for the next send
            # This is complex and depends heavily on server implementation.
            # A simpler persistent connection might involve message framing (e.g., sending message length first)
            # rather than relying on shutdown/EOF.
            # For now, we assume the server handles persistent connections gracefully.
            # We might need to re-establish the socket's write capability if shutdown broke it.
            # Recreating the socket or using a different protocol might be needed if this fails.
            # *** The current shutdown(SHUT_WR) might prevent future sends on some OS/socket implementations ***
            # *** Consider removing shutdown() if the server doesn't require it and uses message framing instead ***
            print("Note: Depending on server implementation and socket behavior after shutdown(SHUT_WR), subsequent sends might fail.")

        else:
            print(f"Received {len(received_data)} bytes from server.")

        # --- Return the raw received data ---
        return pickle.loads(received_data)

    except pickle.PicklingError as e:
        print(f"Error serializing data: {e}")
        return None
    except socket.error as e:
        print(f"Socket error during send/receive: {e}")
        # Consider the connection potentially broken, close it
        _close_connection()
        return None
    except Exception as e:
        print(f"Unexpected error during send/receive: {e}")
        return None

# --- Example Usage (if this file is run directly) ---
if __name__ == "__main__":
    if _client_socket: # Only proceed if connection was successful
        print("\n--- Sending first request ---")
        pre_1 = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4"]
        post_1 = ["g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "e8g8"]
        raw_response_1 = send_moves(pre_1, post_1)

        if raw_response_1:
            print("Raw response 1 received (first 100 bytes):", raw_response_1[:100])
            # --- Optional: Deserialize the response here if needed ---
            try:
                predicted_move_1 = pickle.loads(raw_response_1)
                print(f"Deserialized response 1 (predicted move shape): {predicted_move_1.shape if isinstance(predicted_move_1, np.ndarray) else type(predicted_move_1)}")
            except pickle.UnpicklingError as e:
                print(f"Error unpickling response 1: {e}")
            except Exception as e:
                 print(f"Unexpected error unpickling response 1: {e}")

        print("\n--- Sending second request ---")
        pre_2 = ["d2d4", "d7d5"]
        post_2 = ["c2c4", "e7e6"]
        raw_response_2 = send_moves(pre_2, post_2)

        if raw_response_2:
            print("Raw response 2 received (first 100 bytes):", raw_response_2[:100])
            # --- Optional: Deserialize ---
            try:
                predicted_move_2 = pickle.loads(raw_response_2)
                print(f"Deserialized response 2 (predicted move type): {type(predicted_move_2)}")
            except pickle.UnpicklingError as e:
                print(f"Error unpickling response 2: {e}")
            except Exception as e:
                 print(f"Unexpected error unpickling response 2: {e}")

    else:
        print("\nCannot run example usage: Failed to connect to server initially.")

    # The connection will be closed automatically by atexit when the script ends.
    print("\nClient script finished.")