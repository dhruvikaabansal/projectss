import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Define color ranges in HSV
color_ranges = {
    'blue': ([100, 150, 0], [140, 255, 255]),
    'green': ([40, 70, 70], [80, 255, 255]),
    'red1': ([0, 150, 70], [10, 255, 255]),
    'red2': ([170, 150, 70], [180, 255, 255])
}

# BGR values for drawing colors
bgr_colors = {
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'red': (0, 0, 255)
}

current_color = 'blue'  # Start with blue

points = {
    'blue': [],
    'green': [],
    'red': []
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Prepare mask based on current color
    if current_color == 'red':
        mask1 = cv2.inRange(hsv, np.array(color_ranges['red1'][0]), np.array(color_ranges['red1'][1]))
        mask2 = cv2.inRange(hsv, np.array(color_ranges['red2'][0]), np.array(color_ranges['red2'][1]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower = np.array(color_ranges[current_color][0])
        upper = np.array(color_ranges[current_color][1])
        mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 500:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        points[current_color].append((cx, cy))
        cv2.circle(frame, (cx, cy), 10, bgr_colors[current_color], -1)
        cv2.putText(frame, f"Drawing: {current_color}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, bgr_colors[current_color], 2)
    else:
        points[current_color].append(None)

    # Draw trails for all colors
    for color in points:
        for i in range(1, len(points[color])):
            if points[color][i] is not None and points[color][i-1] is not None:
                cv2.line(frame, points[color][i-1], points[color][i], bgr_colors[color], 5)

    cv2.putText(frame, "Press 'b' for Blue, 'g' for Green, 'r' for Red, 'c' to clear", (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Virtual Paint - Multi Color", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('c'):
        for color in points:
            points[color] = []
    elif key == ord('b'):
        current_color = 'blue'
    elif key == ord('g'):
        current_color = 'green'
    elif key == ord('r'):
        current_color = 'red'

cap.release()
cv2.destroyAllWindows()
