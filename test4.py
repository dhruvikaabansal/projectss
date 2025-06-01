import cv2
import numpy as np
from pyzbar.pyzbar import decode
import csv
from datetime import datetime
import os

# CSV file path
csv_file = "scanned_data.csv"

# If file doesn't exist, create with header
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Scanned_Data", "Time"])

# Load existing data to avoid duplicates
saved_data = set()
with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        if row:
            saved_data.add(row[0])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for barcode in decode(frame):
        data = barcode.data.decode('utf-8')
        pts = barcode.polygon

        # Draw polygon
        if len(pts) > 4:
            hull = cv2.convexHull(np.array([(pt.x, pt.y) for pt in pts], dtype=np.float32))
            hull = np.squeeze(hull).astype(int)
            hull = [(point[0], point[1]) for point in hull]
        else:
            hull = [(pt.x, pt.y) for pt in pts]

        n = len(hull)
        for j in range(n):
            cv2.line(frame, hull[j], hull[(j + 1) % n], (255, 0, 255), 3)

        # Show and save scanned data
        x, y, w, h = barcode.rect
        cv2.putText(frame, data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2)

        if data not in saved_data:
            saved_data.add(data)
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([data, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            print(f"✅ Saved: {data}")
        else:
            print(f"🔁 Already scanned: {data}")

    cv2.imshow("QR/Barcode Scanner + CSV Logger", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
