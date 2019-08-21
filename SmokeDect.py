import cv2
import numpy as np
import time
DEFAULT_VIDEO_SRC = 'test2.avi'

if __name__ == '__main__':
    import sys
    try:
        video_src = sys.argv[1]
    except IndexError:
        video_src = DEFAULT_VIDEO_SRC

    cap = cv2.VideoCapture(video_src)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    frame_count = 1
    while 1:
        ret, frame = cap.read()
        if frame is None:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray_frame = cv2.medianBlur(gray_frame, 5)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 5)
        fmask = fgbg.apply(gray_frame)
        kernel = np.ones((20, 20), np.uint8)
        fmask = cv2.medianBlur(fmask, 3)
        fmask = cv2.dilate(fmask, kernel)

        contours, hierarchy = cv2.findContours(fmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imshow('Smoke Detecion', frame)
        time.sleep(0.03)
        frame_count += 1
        if (cv2.waitKey(10) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
