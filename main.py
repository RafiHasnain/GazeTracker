import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# this function is for finding the mid point
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

font = cv2.FONT_HERSHEY_PLAIN

# This function will return blinking ratio
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # Drawing a green horizontal line
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # Drawing a vertical line
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    # Calculate the horizontal line length
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    # Calculate the vertical line length
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_length / ver_line_length
    return ratio

def get_gaze_ratio( eye_points,facial_landmarks):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                           (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                           (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                           (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                           (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                           (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines( frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    # print( height, width)
    mask = np.zeros((height, width), np.uint8)

    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(grey, grey, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    # This is the grey eyes
    # eye = frame[min_y: max_y, min_x: max_x]
    # grey_eye =cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    grey_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(grey_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    # left side threshold
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    # Count the white part of the left eye
    left_side_white = cv2.countNonZero(left_side_threshold)
    # right side threshold
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    # Count the white part of the right eye
    right_side_white = cv2.countNonZero(right_side_threshold)

    # Top threshold
    top_threshold = threshold_eye[int(height / 2): height, 0: width]
    top_side_white = cv2.countNonZero(top_threshold)
    # bottom threshold
    bottom_threshold = threshold_eye[0: int(height / 2), 0: width]
    bottom_side_white = cv2.countNonZero(bottom_threshold)

    # print("bottom", bottom_side_white)

    try:
        horizontal_gaze_ratio = left_side_white / right_side_white
    except ZeroDivisionError:
        horizontal_gaze_ratio = 0

    try:
        vertical_gaze_ratio = top_side_white / bottom_side_white
    except ZeroDivisionError:
        vertical_gaze_ratio = 0

    # if left_side_white == 0:
    #     gaze_ratio = 1
    # elif right_side_white == 0:
    #     gaze_ratio = 5
    # else:
    #     gaze_ratio = left_side_white / right_side_white

    print("horizontal", horizontal_gaze_ratio)
    print("vertical", vertical_gaze_ratio)
    return horizontal_gaze_ratio


while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(grey)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0),2)
        landmarks = predictor(grey, face)
        # Detect Blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 6:
            cv2.putText(frame, "BLINKING", (50,180), font, 7, (255,0,0),3)


        # Gaze detection
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
        # print(gaze_ratio)

        # Showing Direction

        if gaze_ratio <= 1:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif 1 < gaze_ratio < 1.7:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            new_frame[:] = (255, 0, 0)
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)




        # threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        # eye = cv2.resize(grey_eye, None, fx=5, fy=5)
        # cv2.imshow("Eye", eye) # the grey eye
        # cv2.imshow("Left eye", left_eye) # Only the grey from the black frame
        # cv2.imshow("Threshold", threshold_eye)  # the black and white eye
        # cv2.imshow("left", left_side_threshold)
        # cv2.imshow("right", right_side_threshold)


    cv2.imshow("Frame", frame)
    cv2.imshow("New Frame", new_frame)

    # Press the "S" key to quit
    key = cv2.waitKey(1)
    if key == 27:  # 27 is the the "S" key
        break
cap.release()
cv2.destroyAllWindows()
