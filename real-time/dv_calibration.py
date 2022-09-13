import cv2 as cv
import dv_processing as dv
import numpy as np
import math

# path to the calibration file generate by DV gui
filepath = "../../../DV/calibration/calibration_exp.xml"

if __name__ == "__main__":
    calibration = dv.camera.CalibrationSet.LoadFromFile(filepath)
    camMatrix = calibration.getCameraCalibration("C0")
    print(calibration)
    F = camMatrix.focalLength
    # print("focal length :", camMatrix.focalLength)
    print("F : ", F)
    print("principal point :", camMatrix.principalPoint)
    print("distortion :", camMatrix.distortion)
    # print("master :", camMatrix.master)

    res = camMatrix.resolution
    d = math.sqrt(res[0]**2 + res[1]**2)
    print("cam resolution :", res)
    print("d :", d)
    FOVx = 2 * math.atan(res[0]/(2*F[0]))
    FOVy = 2 * math.atan(res[1]/(2*F[1]))
    print("computed FOV :", math.degrees(FOVx), math.degrees(FOVy))
    
    print(camMatrix.transformationToC0())
    print(calibration.getTransformMatrix())