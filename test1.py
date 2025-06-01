import cv2
import numpy as np

# Webcam open karo
cap = cv2.VideoCapture(0)

# Track karne ke liye color range define karo (Yahan blue color liya hai)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask banao sirf blue color ke liye
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Blue object ke contours find karo
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # chhoti chhoti noise avoid karne ke liye
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            # Object ke center par circle draw karo
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Final frame show karo
    cv2.imshow("Object Tracker", frame)

    # ESC dabane par exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
