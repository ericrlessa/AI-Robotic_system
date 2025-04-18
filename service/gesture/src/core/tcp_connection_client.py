import socket

# Socket Connection
def connect_to_server(ip, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))
    print("Connected to the Raspberry Pi.")
    return client_socket

def send_data(client_socket, data):
    client_socket.send(data)

def check_connection(client_socket):
    try:
        # Check if the socket is still open
        client_socket.send(b'')  # Send an empty message, this doesn't require a response
        return True
    except socket.error:
        return False