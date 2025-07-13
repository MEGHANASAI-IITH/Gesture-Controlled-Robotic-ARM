import cv2
from picamera2 import Picamera2
import mediapipe as mp
import numpy as np
import RPi.GPIO as GPIO
import time

# Initialize Mediapipe and GPIO setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define GPIO pins for servos
servo_pins = [13, 15, 16, 18, 22]  # Define 5 pins for controlling 5 servos
GPIO.setmode(GPIO.BOARD)
for pin in servo_pins:
    GPIO.setup(pin, GPIO.OUT)

# Set up PWM for each servo
servos = [GPIO.PWM(pin, 50) for pin in servo_pins]  # 50 Hz frequency for servo control
for servo in servos:
    servo.start(0)  # Initialize PWM with 0 duty cycle

# Function to calculate angles between joints
joint_list = [[17, 18, 19],  # pinky
              [13, 14, 15],  # ring
              [9, 10, 11],   # middle
              [5, 6, 7]]     # index

def draw_angles(image, results, joint_list):
    if results.multi_hand_landmarks:
        angles = []
        for hand in results.multi_hand_landmarks:
            for joint in joint_list:
                a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])
                b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
                c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])

                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180 / np.pi)
                if angle > 180.0:
                    angle = 360 - angle

                angles.append(angle)
                cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return angles

# Convert angle to duty cycle (for servo movement), with inversion for the little and ring fingers
def angle_to_duty_cycle(angle, invert=False):
    min_dc = 2
    max_dc = 12
    if invert:
        angle = 180 - angle  # Invert the angle for specific fingers
    return min_dc + (max_dc - min_dc) * (angle / 180)

# Setup PiCamera
piCam = Picamera2()
piCam.preview_configuration.main.size = (1280, 720)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()

# Main loop to process camera input and control servos
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    try:
        while True:
            # Capture frame from camera
            image = piCam.capture_array()
            image.flags.writeable = False
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image.flags.writeable = True

            # Draw hand landmarks and calculate joint angles
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
                joint_angles = draw_angles(image, results, joint_list)
                
                # Control servos based on joint angles
                if joint_angles:
                    for i, angle in enumerate(joint_angles):
                        if i < len(servos):  # Limit to available number of servos
                            # Invert the duty cycle for the little finger (first servo) and ring finger (second servo)
                            invert = True if i == 0 or i == 1 else False  # Invert for the first and second servos
                            duty_cycle = angle_to_duty_cycle(angle, invert=invert)
                            servos[i].ChangeDutyCycle(duty_cycle)  # Move corresponding servo

            # Display the camera feed
            cv2.imshow("Hand Gesture Control", image)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("Program stopped")

    finally:
        # Clean up GPIO and close windows
        for servo in servos:
            servo.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()
