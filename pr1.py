import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open camera.")
else:
    ret, frame = cap.read()
    if ret:
        print("✅ Camera is working.")
        cv2.imshow("Test Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Could not read from camera.")
