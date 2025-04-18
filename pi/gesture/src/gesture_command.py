import os
import socket
import sys
from time import sleep

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.motor_control import move_forward, move_backward, turn_left, turn_right, stop, move_diagonal_forward_left, move_diagonal_forward_right, move_diagonal_backward_left, move_diagonal_backward_right

def process_command(command):
    print(f"Received Command: {command}")
    
    if command == "Up":
        move_forward(0.3)
        print("Start Sleep")
        sleep(0.1)
        print("Finished Sleep")
        stop()
    elif command == "Down":
        move_backward(0.3)
        print("Start Sleep")
        sleep(0.1)
        print("Finished Sleep")
        stop()
    elif command == "Left":
        turn_left(0.3)
        print("Start Sleep")
        sleep(0.1)
        print("Finished Sleep")
        stop()
    elif command == "Right":
        turn_right(0.3)
        print("Start Sleep")
        sleep(0.1)
        print("Finished Sleep")
        stop()
    elif command == "Left Up":
        move_diagonal_forward_left(0.3)
        print("Start Sleep")
        sleep(0.1)
        print("Finished Sleep")
        stop()
    elif command == "Right Up":
        move_diagonal_forward_right(0.3)
        print("Start Sleep")
        sleep(0.1)
        print("Finished Sleep")
        stop()
    elif command == "Left Down":
        move_diagonal_backward_left(0.3)
        print("Start Sleep")
        sleep(0.1)
        print("Finished Sleep")
        stop()
    elif command == "Right Down":
        move_diagonal_backward_right(0.3)
        print("Start Sleep")
        sleep(0.1)
        print("Finished Sleep")
        stop()
    elif command == "Stop":
        stop()
    else:
        print("Invalid Command.")