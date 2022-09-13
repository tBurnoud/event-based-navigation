import dv_processing as dv
import cv2 as cv
import argparse
import time 
import numpy as np
import datetime


x, y, w, h = 100, 100, 50, 50 # simply hardcoded the values
track_window = (x, y, w, h)
isInit = False

f1 = dv.RefractoryPeriodFilter((320, 240))
f1.setRefractoryPeriod(datetime.timedelta(milliseconds=250))
f2 = dv.noise.BackgroundActivityNoiseFilter((320, 240))
f2.setBackgroundActivityDuration(datetime.timedelta(milliseconds=1000))

filter_chain = dv.EventFilterChain()
# filter_chain.addFilter(f1)
filter_chain.addFilter(f2)


def apply_camshift(event_slice):
    global track_window
    filter_chain.accept(event_slice)
    filtered = filter_chain.generateEvents()

    vis = dv.visualization.EventVisualizer((320, 240), (75, 75, 75), (0, 0, 255), (255, 0, 0))
    # vis = dv.visualization.EventVisualizer((320, 240), (75, 75, 75), (0, 0, 255), (0, 0, 255))

    event_frame = acc.generateFrame()
    acc.accept(filtered)
    
    img = cv.flip(cv.flip(vis.generateImage(filtered), 0), 1)


    # set up the ROI for tracking
    roi = img[y:y+h, x:x+w]
    hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 100.,29.4)), np.array((180.,255.,255.)))
    roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
        
    cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)


    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    # apply camshift to get the new location
    ret, track_window = cv.CamShift(dst, track_window, term_crit)
    print("bounding rect :", ret)
    # Draw it on image
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv.polylines(img,[pts],True, 255,2)
    
    # cv.imshow('img2',img2)

    cv.imshow("Events", img2)
    cv.waitKey(2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Attempt at camshift on an event stream.')
    args = parser.parse_args()

    camera = dv.io.CameraCapture()

    # Initialize event accumulator with the known resolution
    acc = dv.Accumulator(camera.getEventResolution())

    # Some accumulation parameters
    acc.setMaxPotential(1.0)
    acc.setEventContribution(0.12)
    acc.setRectifyPolarity(False)

    

    # Create the preview window
    cv.namedWindow("Events", cv.WINDOW_NORMAL)

    # Create an event slicer, this will only be used events only camera
    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=15), apply_camshift)
        
    # start read loop
    while True:
        # Get events
        events = camera.getNextEventBatch()

        # If no events arrived yet, continue reading
        if events is None:
            continue

        # rect = dv.boundingRect(events)
        slicer.accept(events)

