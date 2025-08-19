import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os

# Prompt for User Names
user1_name = input("Enter name for Player 1: ")
user2_name = input("Enter name for Player 2: ")

# OpenCV Camera Capture
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# --- Use relative paths instead of absolute ---
BASE_PATH = os.path.join(os.path.dirname(__file__), "Resources")

# Importing all images
imgBackground = cv2.imread(os.path.join(BASE_PATH, "Background.png"))
imgGameOver   = cv2.imread(os.path.join(BASE_PATH, "gameOver.png"))
imgBall       = cv2.imread(os.path.join(BASE_PATH, "Ball.png"), cv2.IMREAD_UNCHANGED)
imgBat1       = cv2.imread(os.path.join(BASE_PATH, "bat1.png"), cv2.IMREAD_UNCHANGED)
imgBat2       = cv2.imread(os.path.join(BASE_PATH, "bat2.png"), cv2.IMREAD_UNCHANGED)

videoPath = os.path.join(BASE_PATH, "StartVideo.mp4")

# Check if images are loaded
if imgBackground is None or imgGameOver is None:
    print("Error: Could not load required images")
    exit()

# Load the start video
startVideo = cv2.VideoCapture(videoPath)
if not startVideo.isOpened():
    print("Error: Could not load the start video")
    exit()

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Game Variables
ballPos = [100, 100]
speedX = 15
speedY = 15
gameOver = False
score = [0, 0]  # Score for Player 1 and Player 2
gameStarted = False  # Game starts only after user presses 's'

# Timer for speed increment
start_time = cv2.getTickCount()

while True:
    if not gameStarted:
        ret, frame = startVideo.read()
        if not ret:  # Restart the video if it ends
            startVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Show the start video with slower speed
        cv2.imshow("Start Video", frame)
        key = cv2.waitKey(40)  # Increase delay to slow down video playback
        if key != -1 and chr(key).lower() == 's':  # Press 's' to start the game (case-insensitive)
            gameStarted = True
        elif key != -1 and chr(key).lower() == 'q':  # Exit on pressing 'q' (case-insensitive)
            print("Exiting game...")
            break
        continue

    _, img = cap.read()
    img = cv2.flip(img, 1)  # Flip image for mirror effect
    imgRaw = img.copy()

    # Check and resize imgBackground if necessary
    if imgBackground.shape != img.shape:
        imgBackground = cv2.resize(imgBackground, (img.shape[1], img.shape[0]))

    # Find the hands and landmarks
    hands, img = detector.findHands(img, flipType=False)

    # Overlay background
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    # Check for hands and handle paddle movement
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)  # Limit paddle movement

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))  # Draw left paddle
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:  # Left paddle collision
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1  # Player 1 scores

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1195, y1))  # Draw right paddle
                if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:  # Right paddle collision
                    speedX = -speedX
                    ballPos[0] -= 30
                    score[1] += 1  # Player 2 scores

    # Game Over Condition (player misses the ball)
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        # Determine the winner (opponent of the one who missed the ball)
        if ballPos[0] < 40:
            winner = user2_name  # Player 2 wins
        else:
            winner = user1_name  # Player 1 wins

        cv2.putText(img, f"Winner: {winner}", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 64, 255), 3)
        cv2.putText(img, f"{user1_name}: {score[0]}", (550, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 64, 255), 2)
        cv2.putText(img, f"{user2_name}: {score[1]}", (550, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 64, 255), 2)
    else:
        # Increment speed every 10 seconds
        current_time = cv2.getTickCount()
        elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
        if elapsed_time > 10:
            speedX += 2 if speedX > 0 else -2
            speedY += 2 if speedY > 0 else -2
            start_time = current_time

        # Move the ball and handle boundary collisions
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball on the screen
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        # Draw scores
        cv2.putText(img, f"{user1_name}: {score[0]}", (300, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 64, 255), 3)
        cv2.putText(img, f"{user2_name}: {score[1]}", (900, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 64, 255), 3)

    # Resize and show the raw image
    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    # Show the final image with overlays
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # Reset the game on pressing 'r'
    if key != -1 and chr(key).lower() == 'r':  # Reset game (case-insensitive)
        ballPos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]
        imgGameOver = cv2.imread(os.path.join(BASE_PATH, "gameOver.png"))

    # Exit the game on pressing 'q'
    if key != -1 and chr(key).lower() == 'q':  # Exit game (case-insensitive)
        print("Exiting game...")
        break
