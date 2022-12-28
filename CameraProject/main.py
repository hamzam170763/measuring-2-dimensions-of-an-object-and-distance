# Importing Libraries
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import imutils
import cv2
from object_detector import *
import numpy as np
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from resize import *
from detect_aruco_type import *
import numpy as np

# Defining Names of Aruco Tags Supported by OPEN CV
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
buildKV = Builder.load_file("Cam.kv")


class CamApp(App):
    # Opening Kivy Window and setting functions on buttons
    imgFrame = None
    corners = None
    detectedImg = None

    def build(self):
        self.detect = 0
        self.measure = 0
        # OpenCV Camera Window using IP Camera
        # img=cv2.imread("custom.jpg")
        #self.capture = cv2.VideoCapture('rtsp://10.135.0.74:8080/h264_ulaw.sdp')
        self.capture = cv2.VideoCapture(0)
        cv2.namedWindow("OpenCV Camera")

        # For Realtime Camera i.e Video
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return screen_manager

    # Getting input from Camera Frame by Frame
    def update(self, *args):
        # Display image from Camera in OpenCV window
        self.imgFrame = self.getFrame()
        # Converting Image to Gray and Showing it in Window
        grayImage = cv2.cvtColor(self.imgFrame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("OpenCV Camera", grayImage)
        # If Detect button is pressed
        if (self.detect == 1):
            # Detecting Aruco Marker
            imgDetected = self.detectAruco()

            if imgDetected is not None:
                self.detectedImg = imgDetected
                print(self.corners)
                # Converting Image to texture to show it in Kivy Window
                buf1 = cv2.flip(imgDetected, 0)
                buf = buf1.tostring()
                texture1 = Texture.create(size=(imgDetected.shape[1], imgDetected.shape[0]), colorfmt='bgr')
                texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                # Display image from the texture
                self.imageWindow.texture = texture1
            # self.detect = 0

        # If Measure button is pressed
        if (self.measure == 1):
            imgMeasured = self.measureSize()
            # Converting Image to texture to show it in Kivy Window
            buf1 = cv2.flip(imgMeasured, 0)
            buf = buf1.tobytes()
            texture1 = Texture.create(size=(imgMeasured.shape[1], imgMeasured.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # Display image from the texture
            self.imageWindow.texture = texture1
            self.measure = 0
            self.detect = 0

    # Getting the frame when button is clicked
    def getFrame(self):
        ret, frame = self.capture.read()
        return frame

    # Detect Btn Press
    def detectTrue(self, instance):
        instance.parent.ids.measureBTN.disabled = False
        self.imageWindow = instance.parent.ids.image
        self.detect = 1
        print(self.detect)

    # Measure Btn Press
    def measureTrue(self, instance):
        if self.imageWindow.texture != None:
            self.imageWindow = instance.parent.ids.image
            self.measure = 1
            print(self.measure)

    # Detecting Aruco Marker
    def detectAruco(self):
        # load the input image from disk and resize it
        # image = cv2.imread("test.jpeg")
        print(self.imgFrame)
        # imgDetected = imutils.resize(self.imgFrame, width=600, height=480)
        imgDetected = self.imgFrame
        # loop over the types of ArUco dictionaries
        for (arucoName, arucoDict) in ARUCO_DICT.items():
            # load the ArUCo dictionary, grab the ArUCo parameters, and
            # attempt to detect the markers for the current dictionary
            arucoDict = cv2.aruco.Dictionary_get(arucoDict)
            arucoParams = cv2.aruco.DetectorParameters_create()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(imgDetected, arucoDict, parameters=arucoParams)
            print(corners)
            # if at least one ArUco marker was detected display the ArUco
            # name to our terminal
            if len(corners) > 0:
                self.corners = corners
                print("[INFO] detected {} markers for '{}'".format(len(corners), arucoName))
                # flatten the ArUco IDs list
                ids = ids.flatten()
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
                    # draw the bounding box of the ArUCo detection
                    cv2.line(imgDetected, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(imgDetected, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(imgDetected, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(imgDetected, bottomLeft, topLeft, (0, 255, 0), 2)
                    # compute and draw the center (x, y)-coordinates of the ArUco
                    # marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    cv2.circle(imgDetected, (cX, cY), 4, (0, 0, 255), -1)
                    # draw the ArUco marker ID on the image
                    cv2.putText(imgDetected, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0), 2)
                    print("[INFO] ArUco marker ID: {}".format(markerID))
                    print("[INFO] Aruco marker size : {}".format(ARUCO_DICT[arucoName]))
                    print(type(arucoName))
                    print(arucoName)
                    print(type(ARUCO_DICT))
                    # show the output image
                # cv2.imshow("Aruco Detected", imgDetected)
                # cv2.waitKey(0)

                return imgDetected

    # Measure Size of Object
    def measureSize(self):
        # Load Object Detector
        detector = HomogeneousBgDetector()
        # Load Image
        img = self.detectedImg
        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(self.corners[0], True)
        print("Aruco Perimeter is ", aruco_perimeter)
        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 40

        contours = detector.detect_objects(img)

        # Draw objects boundaries
        for cnt in contours:
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            print("X is:", x)
            print("Y is:", y)
            print("W is:", w)
            print("H is:", h)

            # Get Width and Height of the Objects by applying the Ratio pixel to cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 100)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y - 70)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

        return img

    def calibration(self, instance):
        # input image for calibration
        calibrated = False
        # will be calculated after calibration
        focal_len = 0
        # Aruco detector
        detector = Detect_aruco()
        img = cv2.imread("9in.jpg")
        # Resizing the image to a certain percent for less computation
        image = self.resize_image(img)
        #image=img
        # These Measures are in "cm"
        self.KNOWN_DISTANCE = float(instance.parent.ids.knownDistance.text)
        self.KNOWN_WIDTH = float(instance.parent.ids.knownWidth.text)

        # Test Case
        # KNOWN_DISTANCE = 22.86 #cm  in inches  9
        # KNOWN_WIDTH = 17.5 #cm  in inches  6.9

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
            # This is the formula for focal length
            self.focal_len = (p_width * self.KNOWN_DISTANCE) / self.KNOWN_WIDTH
            print("[INFO]Focal Length Calculated")
            print("Focal Length is:{}".format(self.focal_len))
        if self.focal_len > 0:
            calibrated = True
        return calibrated, self.focal_len, self.KNOWN_WIDTH

    def measure_distance(self, instance):
        img = cv2.imread("1.5ft.jpg")
        # Aruco Detector
        detector = Detect_aruco()
        # Resizing
        image = self.resize_image(img)
        #image=img
        # Detected aruco parameters
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
            # Drawing boxes to get the box on the detected aruco
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0),
                        2)
            #Pixel width of aruco detected
            p_width = np.linalg.norm(np.asarray(topRight) - np.asarray(topLeft))
            # This is same formula of focal length used in calibration but rearranged
            distance = self.distance_to_camera(self.KNOWN_WIDTH, self.focal_len, p_width)
            # Formatted Distance
            form_distance = "{:.2f}".format(distance)
            print("[INFO] Distance calculated is {} cm".format(form_distance))
            # draw the ArUco marker ID on the image
            cv2.putText(image, "Distance is " + str(form_distance) + "cm", (cX - 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            self.imageWindow = instance.parent.ids.image
            buf1 = cv2.flip(image, 0)
            buf = buf1.tobytes()
            texture1 = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # Display image from the texture
            self.imageWindow.texture = texture1
            # show the output image
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)

    def resize_image(self,img):
        # print('Original Dimensions : ', img.shape)

        scale_percent = 15  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        # dimensions of resized image
        dim = (width, height)
        # Resizing
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # print('Resized Dimensions : ', resized.shape)
        # return resized image
        return resized

    def distance_to_camera(self,knownWidth, focalLength, perWidth):
        # compute and return the distance from the maker to the camera
        return (float(knownWidth) * float(focalLength)) / float(perWidth)

class MyBg(Screen):
    pass


class MeasureDimensions(Screen):
    pass


class Calibrate(Screen):
    pass


class MeasureDistance(Screen):
    pass


screen_manager = ScreenManager()
screen_manager.add_widget(MyBg(name="MyBg"))
screen_manager.add_widget(MeasureDimensions(name="Dimensions"))
screen_manager.add_widget(Calibrate(name="Calibrate"))
screen_manager.add_widget(MeasureDistance(name="Distance"))

if __name__ == '__main__':
    CamApp().run()
