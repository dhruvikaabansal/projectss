import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Blue color range (HSV)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# List to store points where object moves (trail)
points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip frame horizontally (mirror view)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 500:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        points.append((cx, cy))  # Add current position to points list

        # Draw circle on current position
        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)
        cv2.putText(frame, "Drawing", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    else:
        # If object not detected, add None to maintain trail continuity
        points.append(None)

    # Draw trail lines between points
    for i in range(1, len(points)):
        if points[i] is not None and points[i-1] is not None:
            cv2.line(frame, points[i-1], points[i], (255, 0, 0), 5)

    cv2.imshow("Virtual Paint", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('c'):  # Press 'c' to clear drawing
        points = []

cap.release()
cv2.destroyAllWindows()
