import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for barcode in decode(frame):
        data = barcode.data.decode('utf-8')
        pts = barcode.polygon

        # Draw polygon around barcode/QR
        if len(pts) > 4:
            hull = cv2.convexHull(np.array([pt for pt in pts], dtype=np.float32))
            hull = [(int(point[0]), int(point[1])) for point in np.squeeze(hull)]
        else:
            hull = [(pt.x, pt.y) for pt in pts]  # Make sure it's list of (int, int)

        n = len(hull)
        for j in range(0, n):
            cv2.line(frame, hull[j], hull[(j + 1) % n], (255, 0, 255), 3)

        # Show data
        x, y, w, h = barcode.rect
        cv2.putText(frame, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2)
        print(f"Scanned Data: {data}")

    cv2.imshow("QR/Barcode Scanner", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
