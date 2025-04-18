from gpiozero import Motor, Servo
from time import sleep

# 2 L298N Motor Driver GPIO pins for 4 DC motors
M1_A = 16  # Back Right Motor, IN2
M1_B = 12  # Back Right Motor, IN1
M2_A = 20  # Back Left Motor, IN4
M2_B = 21  # Back Left Motor, IN3
M3_A = 10  # Front Right Motor, IN4
M3_B = 22  # Front Right Motor, IN3
M4_A = 17  # Front Left Motor, IN1
M4_B = 27  # Front Left Motor, IN2

# SG90 Servo for camera Y-axis movement
SERVO_PIN = 14  # GPIO pin for the SG90 servo

# Setup DC motors using gpiozero Motor class
motor1 = Motor(M1_A, M1_B)
motor2 = Motor(M2_A, M2_B)
motor3 = Motor(M3_A, M3_B)
motor4 = Motor(M4_A, M4_B)

# Setup SG90 servo using gpiozero Servo class

def set_motor_speed(motor, speed):
    """
    Sets the speed and direction of a motor.
    speed: -1 (reverse) to 1 (forward), 0 (stop)
    """
    if speed > 0:
        motor.forward(speed)
    elif speed < 0:
        motor.backward(abs(speed))
    else:
        motor.stop()

def move_robot(vx, vy, omega):
    """
    Controls the robot's movement based on desired velocities in X, Y, and angular (omega).
    vx: velocity along the X-axis (right/left)
    vy: velocity along the Y-axis (forward/backward)
    omega: angular velocity (rotation)
    """
    # Calculate the motor speeds based on the inputs
    s1 = vy - vx - omega  # Motor 1 (front-left)
    s2 = vy + vx + omega  # Motor 2 (front-right)
    s3 = vy + vx - omega  # Motor 3 (back-left)
    s4 = vy - vx + omega  # Motor 4 (back-right)
    
    # Apply motor speeds
    set_motor_speed(motor1, s1)
    set_motor_speed(motor2, s2)
    set_motor_speed(motor3, s3)
    set_motor_speed(motor4, s4)

# Define movement functions for omnidirectional control

def move_robot(vx, vy, omega):
    """
    Controls the robot's movement based on desired velocities in X, Y, and angular (omega).
    vx: velocity along the X-axis (right/left)
    vy: velocity along the Y-axis (forward/backward)
    omega: angular velocity (rotation)
    """
    # Calculate the motor speeds based on the inputs
    s1 = vy - vx - omega  # Motor 1 (front-left)
    s2 = vy + vx + omega  # Motor 2 (front-right)
    s3 = vy + vx - omega  # Motor 3 (back-left)
    s4 = vy - vx + omega  # Motor 4 (back-right)
    
    # Apply motor speeds
    set_motor_speed(motor1, s1)
    set_motor_speed(motor2, s2)
    set_motor_speed(motor3, s3)
    set_motor_speed(motor4, s4)

"""
Moves the robot forward at a specified speed.
speed: 0 to 1 (for Motor control)
"""
def move_forward(speed):
    set_motor_speed(motor1, speed)
    set_motor_speed(motor2, speed)
    set_motor_speed(motor3, speed)
    set_motor_speed(motor4, speed)

def move_backward(speed):
    set_motor_speed(motor1, -speed)
    set_motor_speed(motor2, -speed)
    set_motor_speed(motor3, -speed)
    set_motor_speed(motor4, -speed)

def turn_left(speed):
    set_motor_speed(motor1, -speed)
    set_motor_speed(motor2, speed)
    set_motor_speed(motor3, -speed)
    set_motor_speed(motor4, speed)

def turn_right(speed):
    set_motor_speed(motor1, speed)
    set_motor_speed(motor2, -speed)
    set_motor_speed(motor3, speed)
    set_motor_speed(motor4, -speed)

def stop():
    motor1.stop()
    motor2.stop()
    motor3.stop()
    motor4.stop()

# Additional functions for omnidirectional movement:

def move_sideways_left(speed):
    set_motor_speed(motor1, -speed)
    set_motor_speed(motor2, speed)
    set_motor_speed(motor3, speed)
    set_motor_speed(motor4, -speed)

def move_sideways_right(speed):
    set_motor_speed(motor1, speed)
    set_motor_speed(motor2, -speed)
    set_motor_speed(motor3, -speed)
    set_motor_speed(motor4, speed)

def move_diagonal_forward_left(speed):
    set_motor_speed(motor1, 0)
    set_motor_speed(motor2, speed)
    set_motor_speed(motor3, speed)
    set_motor_speed(motor4, 0)

def move_diagonal_forward_right(speed):
    set_motor_speed(motor1, speed)
    set_motor_speed(motor2, 0)
    set_motor_speed(motor3, 0)
    set_motor_speed(motor4, speed)

def move_diagonal_backward_left(speed):
    set_motor_speed(motor1, 0)
    set_motor_speed(motor2, -speed)
    set_motor_speed(motor3, -speed)
    set_motor_speed(motor4, 0)

def move_diagonal_backward_right(speed):
    set_motor_speed(motor1, -speed)
    set_motor_speed(motor2, 0)
    set_motor_speed(motor3, 0)
    set_motor_speed(motor4, -speed)

def control_camera_y_angle(angle):
    """
    Controls the Y-axis angle of the camera.
    angle: -1 to 1 (for Servo control)
    """
    servo = Servo(SERVO_PIN)
    servo.value = (angle / 180) * 2 - 1  # Map 0-180 degrees to -1 to 1 scale
    sleep(0.5)  # Delay to let the servo move

def cleanup():
    """ Clean up GPIO pins after the program finishes """
    stop()
    sleep(1)  # Give time for motors to stop before cleanup
    # gpiozero handles cleanup automatically when the script ends
