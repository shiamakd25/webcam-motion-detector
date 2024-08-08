import cv2
import time
from emails import send_email
import glob
from threading import Thread

video = cv2.VideoCapture(0)
time.sleep(1)

initial_frame = None
status_list = []
count = 0

while True:
    status = 0
    check, frame = video.read()
    cv2.imwrite(f"images/{count}.png", frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if initial_frame is None:
        initial_frame = gray_frame_gau

    delta_frame = cv2.absdiff(initial_frame, gray_frame_gau)
    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        if rectangle.any():
            status = 1
            count += 1
            cv2.imwrite(f"images/{count}.png", frame)
            cap_images = glob.glob("images/*.png")
            index = int(len(cap_images) / 2)
            captured_image = cap_images[index]
            
    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[0] == 1 and status_list[1] == 0:
        email_thread = Thread(target=send_email)
        email_thread.daemon = True
        email_thread.start()

    cv2.imshow("Video", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()