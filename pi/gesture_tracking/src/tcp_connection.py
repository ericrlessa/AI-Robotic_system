import socket

def wait_for_client_connection():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind(('0.0.0.0', 12000))
        server_socket.listen(1)
        print("Waiting for a connection...")

        client_socket, addr = server_socket.accept()
        print(f"Connected to {addr}")
        return client_socket, addr

    except socket.error as e:
        print(f"Socket error: {e}")
        return None, None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

    finally:
        server_socket.close()

def receive_command(client_socket):
    try:
        data = client_socket.recv(1024)
        if not data:
            print("Client disconnected.")
            return None
        return data.decode().strip()

    except ConnectionResetError:
        print("Connection reset by peer.")
        return None

    except socket.timeout:
        print("Socket timed out while waiting for data.")
        return None

    except socket.error as e:
        print(f"Socket error: {e}")
        return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
