import sys
import os
import time
from gesture_command import process_command, stop
from tcp_connection import wait_for_client_connection, receive_command
import argparse
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from cloud_service.aws.pubsub_aws_iot import subscribe

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.motor_control import move_robot

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cloud", default="Y")
    args = parser.parse_args()

    if args.cloud == 'Y':
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
    mqtt_connection = subscribe(client_id="pi_robot", callback=process_command_json)
    while(True):
        time.sleep(0.1)

def process_command_json(data_str):
    try:
        data = json.loads(data_str)  # convert JSON string to Python dict
        command = data.get("command")

        if command == "tracking":
            vx = data.get("vx", 0)
            vy = data.get("vy", 0)
            omega = data.get("omega", 0)
            move_robot(vx, vy, omega)
            time.sleep(0.1)
            stop()

        elif command == "gesture":
            gesture = data.get("gesture", "unknown")
            process_command(gesture)

        else:
            print(f"[ERROR] Unknown command: {command}")

    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON: {e}")

if __name__ == "__main__":
    main()