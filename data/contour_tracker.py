import dv_processing as dv
from scipy.spatial.transform import Rotation
import cv2 as cv
import datetime
import pandas as pd
import numpy as np
import math


clustersColor = [[255, 0, 0], [0, 255, 0], [255, 255, 0], [255, 0, 255], 
                 [0, 255, 255], [125, 125, 125], [75, 75, 125], [125, 75, 75], [75, 125, 75]]




#camera intrinsict matrix
camMatrix = np.array([[3.7453954539334285e+02, 0., 1.5282441586920595e+02],
                      [0, 3.7616000309566266e+02, 1.1664521377777844e+02],
                      [0., 0., 1.]])


# camTransfo = np.array([[.0], [0.875], [3.58]])
camTransfo = np.array([[0.], [0.], [0.]])

#camera rotation
camRota = Rotation.from_euler('y', -20, degrees=True).as_matrix()

#extrinsict camera matrix
extraMat = np.hstack((camRota, camTransfo)) 

invMatrix = np.linalg.inv(camMatrix)


#camera parameters 
principalPoint = np.array([160, 120])
# principalPoint = np.array([0., 0.])
fxy = 0.02
F = 375e-3 

drone_size = (0.36, 0.14)

class ContourTracker:
    def __init__(self, downFactor=0.66, threshold=165):

        self.position = []

        self.proj = np.array([0., 0., 0.])
        self.beta = 0.75

        self.downsizeFactor = downFactor

        self.x, self.y, self.w, self.h = 160, 120, 30, 30 # simply hardcoded the values
        self.track_window = (self.x, self.y, self.w, self.h)

        self.noiseFilter = dv.noise.FastDecayNoiseFilter((320, 240), subdivisionFactor=8, noiseThreshold=8)
        self.noiseFilter.setHalfLife(datetime.timedelta(milliseconds=250))

        #event accumulator
        self.acc = dv.Accumulator((320, 240))
        self.acc.setMaxPotential(1.0)
        self.acc.setEventContribution(0.12)

        #event visualiaser
        self.vis = dv.visualization.EventVisualizer((320, 240), (175, 175, 175))

        self.prevDepth = 0
        self.alpha = 0.75

        self.threshold = threshold

    def apply_meanShift(self, img):
        img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        roi = img[self.y:self.y+self.h, self.x:self.x+self.w]
        hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        # hsv_roi = roi
        hsv_green = cv.cvtColor(np.uint8([[[0,255,0 ]]]),cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((0.,0.,100.)))
        mask2 = cv.inRange(hsv_roi, np.array((50., 255., 255.)), np.array((70.,255.,255.)))
        # mask = cv.inRange(hsv_roi, np.array((75, 0, 75)), np.array((255, 0, 255)))
        # cv.imshow("mask", hsv_roi)
        roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
            
        cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
        term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)


        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,500],1)

        # apply camshift to get the new location
        ret, self.track_window = cv.CamShift(gray, self.track_window, term_crit)
        # print("bounding rect :", ret)
        # print("tracking window : ", track_window)

        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        cv.polylines(img,[pts],True, 255,2)

        # cv.imshow("cv", img)
        # cv.waitKey(15)

        return ret

    def track(self, event_slice):
        #processing the events through the event filter
        self.noiseFilter.accept(event_slice)
        filtered = self.noiseFilter.generateEvents()

        #getting the everage time of the current batch
        time = (filtered.getLowestTime() + filtered.getHighestTime()) / 2

        #recording was done with camera flipped upside down so we need to un-flip it first
        img = cv.flip(cv.flip(self.vis.generateImage(filtered), 0), 1)
        img = cv.resize(img, (int(320*self.downsizeFactor), int(240*self.downsizeFactor)))

        black = np.zeros((240, 320), dtype=np.uint8)

        canny_input = img
        canny_output = cv.Canny(canny_input, self.threshold, self.threshold*2)

        contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        a = int(20*self.downsizeFactor)
        b =int(320*self.downsizeFactor)
        ROI = black[0:a, 0:b]
        canny_output[0:a, 0:b] = ROI
        
        a = int(220*self.downsizeFactor)
        b = int(230*self.downsizeFactor)
        c = int(172*self.downsizeFactor)
        d = int(182*self.downsizeFactor)
        ROI = black[a:b, c:d]
        canny_output[a:b, c:d] = ROI

        boundingBox = self.apply_meanShift(canny_output)
        # print(boundingBox)
        pts = cv.boxPoints(boundingBox)
        pts = np.int0(pts)
        cv.polylines(canny_input,[pts],True, 255,2)
        area = boundingBox[1][0] * boundingBox[1][1]
        px = []
        py = []
        for p in pts:
            px.append(p[0])
            py.append(p[1])
        min_x = np.min(px)
        max_x = np.max(px)
        min_y = np.min(py)
        max_y = np.max(py)

        size = (boundingBox[1][1], boundingBox[1][0])
        # print("size :", size)

        if area > 50:
        # if size[0] != 0 and size[1] != 0:
            depth = (F * drone_size[0]) / (size[0] * fxy)
            # depth = ((F * drone_size[0]) / (size[0] * fxy)) + ((F * drone_size[1]) / (size[1] * fxy))
            # depth /= 2

            # print("depth", depth)

            prevDepth = self.alpha * depth + (1-self.alpha) * self.prevDepth
            c = np.array([boundingBox[0][0] - principalPoint[0], boundingBox[0][1] - principalPoint[1], 1.0])

            cX = (c.transpose() @ invMatrix) * 1
            wX = extraMat @ np.array([cX[0], cX[1], prevDepth, 1.0])
            self.proj = (wX * self.beta) + self.proj * (1-self.beta)
            if self.proj[2] > 0:
                self.position.append([time, self.proj[0], self.proj[1], self.proj[2]])
            # cv.circle(gray_ev, (int(c[0]), int(c[1])), 5, (200, 100, 100))


        # cv.imshow("ev", gray_ev)
        # cv.waitKey(2)

    def get_positions(self):
        return pd.DataFrame(self.position)

    def write_position(self):
        df = pd.DataFrame(self.position)
        df.columns = ["time", "x", "y", "z"]
        if not df.empty:
            df.to_csv("/home/tommy/workspace/dv-processing/data/feature_position.csv")
