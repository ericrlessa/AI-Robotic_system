import sys
import os
import time
from gesture_command import process_command
from tcp_connection import wait_for_client_connection, receive_command

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from cloud.aws.pubsub_aws_iot import subscribe

def main():
    value = os.getenv('CLOUD')
    if value and value == 'Y':
        aws_connection()
    else:
        socket_connection()

def socket_connection():
    client_socket = None  # Initialize client_socket to ensure it's defined before use
    command = None
    
    try:
        while True:
            try:
                # Only try to get a connection if the client_socket is not initialized
                if client_socket is None or command is None:
                    client_socket, _ = wait_for_client_connection()
                    if client_socket is None:
                        continue  # Skip the rest of the loop if connection fails

                command = receive_command(client_socket)
                if command is not None:
                    process_command(command)
            except Exception as e:
                print(f"Error processing command: {e}")
                
    finally:
        if client_socket:  # Ensure client_socket is only closed if it was opened
            client_socket.close()

def aws_connection():
    mqtt_connection = subscribe(client_id="pi_robot", callback=process_command)
    while(True):
        time.sleep(0.1)

if __name__ == "__main__":
    main()