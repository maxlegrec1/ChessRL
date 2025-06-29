import socket
import pickle
import numpy as np
# --- Configuration ---
SERVER_HOST = '127.0.0.1'  # Doit correspondre à l'hôte du serveur
SERVER_PORT = 65432        # Doit correspondre au port du serveur
BUFFER_SIZE = 4096

# --- Données à envoyer ---
pre_pos = ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4"]
post_pos = ["g8f6","e1g1","f8e7","f1e1","b7b5","a4b3","e8g8"]
# Vous pouvez changer cette liste pour tester différents scénarios

# --- Sérialisation des données ---
try:
    data_to_send_pickle = pickle.dumps((pre_pos,post_pos))
except pickle.PicklingError as e:
    print(f"Erreur lors de la sérialisation des données: {e}")
    exit()
except Exception as e:
    print(f"Erreur inattendue lors de la sérialisation: {e}")
    exit()


print(f"Tentative de connexion au serveur {SERVER_HOST}:{SERVER_PORT}...")
# --- Connexion et communication ---
try:
    # Utiliser 'with' garantit que le socket sera fermé même en cas d'erreur
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((SERVER_HOST, SERVER_PORT))
        print("Connecté au serveur.")

        # --- Envoi des données ---
        print(f"Envoi des données ({len(data_to_send_pickle)} octets): {(pre_pos,post_pos)}")
        client_socket.sendall(data_to_send_pickle)
        print("Données envoyées. Attente de la réponse...")

        # Optionnel mais recommandé: Indiquer qu'on a fini d'envoyer.
        # Cela aide le serveur à savoir quand recv() doit s'arrêter si le serveur
        # lit jusqu'à la fin du flux (EOF).
        client_socket.shutdown(socket.SHUT_WR)

        # --- Réception de la réponse ---
        received_data = b""
        while True:
            chunk = client_socket.recv(BUFFER_SIZE)
            if not chunk:
                break # Le serveur a fermé la connexion après avoir envoyé sa réponse
            received_data += chunk

        if not received_data:
            print("Aucune réponse reçue du serveur.")
        else:
            print(f"Reçu {len(received_data)} octets du serveur.")
            # --- Désérialisation de la réponse ---
            try:
                predicted_move = pickle.loads(received_data)
                print("-" * 30)
                print(f"Coup prédit reçu du serveur: {predicted_move.shape}")
                print("-" * 30)
            except pickle.UnpicklingError as e:
                print(f"Erreur de désérialisation Pickle de la réponse: {e}")
                # Essayer de décoder comme une chaîne pour voir si c'est un message d'erreur
                try:
                    error_message = received_data.decode('utf-8', errors='ignore')
                    print(f"Réponse brute (potentiel message d'erreur texte): {error_message}")
                except:
                     print("Impossible de décoder la réponse brute.")
            except Exception as e:
                 print(f"Erreur inattendue lors de la désérialisation de la réponse: {e}")


except socket.error as e:
    print(f"Erreur de socket lors de la connexion/communication: {e}")
except ConnectionRefusedError:
    print(f"Échec de la connexion: Le serveur ({SERVER_HOST}:{SERVER_PORT}) est-il lancé et accessible?")
except Exception as e:
    print(f"Une erreur inattendue est survenue: {e}")

print("Client terminé.")