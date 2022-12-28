import cv2
from resize import *
from detect_aruco_type import *
import numpy as np


# Calibration

class Measurement:

    def __init__(self):
        pass

    def calibration(self):
        # input image for calibration
        calibrated = False
        focal_len = 0
        # Aruco detector
        detector = Detect_aruco()
        img = cv2.imread("9in.jpg")
        image = Measurement.resize_image(img)

        # These Measures are in "cm"
        KNOWN_DISTANCE = float(input("Enter the marker distance :"))
        KNOWN_WIDTH = float(input("Enter the width of marker :"))

        # Test Case
        # KNOWN_DISTANCE = 8.7 #in 22.86 in cm
        # KNOWN_WIDTH = 6.9 #in 17 in cm

        # Detecting aruco
        arucoName, corners, ids = detector.detecting_aruco(image)
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            # compute and draw the center (x, y)-coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            # computing pixel width using Euclidean Distance
            p_width = np.linalg.norm(np.asarray(topRight) - np.asarray(topLeft))
            focal_len = (p_width * KNOWN_DISTANCE) / KNOWN_WIDTH
            print("[INFO]Focal Length Calculated")
            print("Focal Length is:{}".format(focal_len))
        if focal_len > 0:
            calibrated = True
        return calibrated, focal_len, KNOWN_WIDTH


    def measure_distance(focal_len, KNOWN_WIDTH):
        img = cv2.imread("113in-r.jpg")
        detector = Detect_aruco()
        # Resizing
        image = Measurement.resize_image(img)
        arucoName, corners, ids = detector.detecting_aruco(image)
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            # compute and draw the center (x, y)-coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)
            p_width = np.linalg.norm(np.asarray(topRight) - np.asarray(topLeft))
            distance = Measurement.distance_to_camera(KNOWN_WIDTH, focal_len, p_width)
            form_distance = "{:.2f}".format(distance)
            print("[INFO] Distance calculated is {} in".format(form_distance))
            # draw the ArUco marker ID on the image
            cv2.putText(image, "Distance is "+str(form_distance)+"in", (cX-10, cY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # show the output image
            cv2.imshow("Image", image)
            cv2.waitKey(0)




    def resize_image(img):
        #print('Original Dimensions : ', img.shape)

        scale_percent = 15  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        # dimensions of resized image
        dim = (width, height)
        # Resizing
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        #print('Resized Dimensions : ', resized.shape)
        # return resized image
        return resized


    def distance_to_camera(knownWidth, focalLength, perWidth):
        # compute and return the distance from the maker to the camera
        return (float(knownWidth) * float(focalLength)) / float(perWidth)



#img = cv2.imread("images/9in.jpg")
calibrated, focal_len, KNOWN_WIDTH = Measurement().calibration()
if calibrated == True:
    print("[INFO] System is Calibrated")
    Measurement.measure_distance(focal_len, KNOWN_WIDTH)
