import socket
import pickle
import atexit
import struct # Import struct for packing/unpacking the length

# ... (Keep SERVER_HOST, PORT, BUFFER_SIZE, _client_socket, _close_connection, atexit registration, initial connection logic) ...

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
# --- Function to send moves and receive response (with Length Prefixing) ---

def connect_to_server():
    """
    Connects to the server using the global socket variable.
    This function is called when the module is imported.
    """
    global _client_socket
    try:
        print(f"Attempting to connect to server {SERVER_HOST}:{SERVER_PORT}...")
        _client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _client_socket.connect((SERVER_HOST, SERVER_PORT))
        print("Successfully connected to the server.")
    except socket.error as e:
        print(f"FATAL: Failed to connect to server: {e}")
        _client_socket = None

def send_moves(policies):
    """
    Sends pre_pos and post_pos lists to the connected server using length-prefix framing
    and returns the raw response data (also expected to be length-prefixed).

    Args:
        pre_pos (list): A list of previous position strings.
        post_pos (list): A list of post position strings.

    Returns:
        bytes: The raw payload data received from the server, or None if an error occurred.
               Returns b'' if server sends an empty message (length 0).
    """
    global _client_socket

    if not _client_socket:
        print("Error: No active connection to the server.")
        return None

    try:
        # --- Serialize the data ---
        payload = pickle.dumps(policies)
        payload_length = len(payload)

        # --- Pack the length into a fixed-size header (e.g., 8 bytes, unsigned long long) ---
        # 'Q' means unsigned long long (8 bytes), Network byte order '>'
        header = struct.pack('>Q', payload_length)

        # --- Send header first, then payload ---
        #print(f"Client: Sending header ({len(header)} bytes, length={payload_length})")
        _client_socket.sendall(header)
        #print(f"Client: Sending payload ({payload_length} bytes)")
        _client_socket.sendall(payload)
        print("Client: Data sent. Waiting for response...")

        # --- Receive the response header ---
        header_size = struct.calcsize('>Q') # Should be 8
        print(f"Client: Receiving response header ({header_size} bytes)...")
        response_header = b""
        while len(response_header) < header_size:
            chunk = _client_socket.recv(header_size - len(response_header))
            if not chunk:
                #print("Client: Error: Connection closed while receiving response header.")
                _close_connection()
                return None
            response_header += chunk

        response_payload_length = struct.unpack('>Q', response_header)[0]
        print(f"Client: Received response header. Expecting payload of {response_payload_length} bytes.")

        # --- Receive the response payload ---
        response_payload = b""
        while len(response_payload) < response_payload_length:
            bytes_to_read = min(BUFFER_SIZE, response_payload_length - len(response_payload))
            chunk = _client_socket.recv(bytes_to_read)
            if not chunk:
                #print("Client: Error: Connection closed while receiving response payload.")
                _close_connection()
                return None
            response_payload += chunk

        print(f"Client: Received complete response payload ({len(response_payload)} bytes).")
        # --- Return the raw *payload* data ---
        # The caller will be responsible for unpickling this raw payload
        return response_payload

    except pickle.PicklingError as e:
        #print(f"Client: Error serializing data: {e}")
        return None
    except (struct.error, socket.error) as e:
        #print(f"Client: Socket or struct error during send/receive: {e}")
        _close_connection() # Connection is likely broken
        return None
    except Exception as e:
        #print(f"Client: Unexpected error during send/receive: {e}")
        # Optionally close connection here too if the error seems connection-related
        # _close_connection()
        return None

# ... (Keep the rest of the client file, including the example usage block if needed) ...
# Remember that the *caller* of send_moves now receives the raw *payload*
# and needs to do pickle.loads() on it.
# Example modification in the calling script:
# raw_payload = client_final.send_moves(pre_moves, post_moves)
# if raw_payload is not None:
#    try:
#        prediction = pickle.loads(raw_payload)
#        print("Deserialized prediction:", prediction)
#    except pickle.UnpicklingError:
#        print("Failed to deserialize payload.")