import numpy as np
import cv2
import glob
import argparse
import sys
from monoCalibration import monoCalib
from calibrationStore import loadCoefficients, saveStereoCoefficients

# 终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_size = None


def stereoCalibrate(paramL, paramR, dirL, prefixL, dirR, prefixR, imageFormat, saveFile, squareSize, width=8, height=5):
    """ 双目标定和校正 """
    objp, leftp, rightp = load_image_points(dirL, prefixL, dirR, prefixR, imageFormat, squareSize, width, height)
    
    # 获取单目标定参数
    if paramL and paramR: # 从文件中读取
        K1, D1 = loadCoefficients(paramL)
        K2, D2 = loadCoefficients(paramR)
    else: # 直接利用图片进行标定
        _,K1, D1, _, _ = monoCalib(dirPath=dirL, prefix=prefixL, imageFormat=imageFormat, saveFile='stereoCalibParamL.yml', squareSize=squareSize, width=width, height=height)
        _,K2, D2, _, _ = monoCalib(dirPath=dirR, prefix=prefixR, imageFormat=imageFormat, saveFile='stereoCalibParamR.yml', squareSize=squareSize, width=width, height=height)

    flag = 0
    # flag |= cv2.CALIB_FIX_INTRINSIC
    flag |= cv2.CALIB_USE_INTRINSIC_GUESS
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objp, leftp, rightp, K1, D1, K2, D2, image_size)
    print("Stereo calibration rms: ", ret)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

    saveStereoCoefficients(saveFile, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q, ret)


def load_image_points(dirL, prefixL, dirR, prefixR, imageFormat, squareSize,width=8, height=5):
    global image_size
    pattern_size = (width, height)  # Chessboard size!
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * squareSize  # Create real world coords. Use your metric.

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpointsL = []  # 2d points in image plane.
    imgpointsR = []  # 2d points in image plane.

    # Left directory path correction. Remove the last character if it is '/'
    if dirL[-1:] == '/':
        dirL = dirL[:-1]

    # Right directory path correction. Remove the last character if it is '/'
    if dirR[-1:] == '/':
        dirR = dirR[:-1]

    # Get images for left and right directory. Since we use prefix and formats, both image set can be in the same dir.
    left_images = glob.glob(dirL + '/' + prefixL + '*.' + imageFormat)
    right_images = glob.glob(dirR + '/' + prefixR + '*.' + imageFormat)   # Images should be perfect pairs. Otherwise all the calibration will be false.
    # Be sure that first cam and second cam images are correctly prefixed and numbers are ordered as pairs.
    # Sort will fix the globs to make sure.
    left_images.sort()
    right_images.sort()

    # Pairs should be same size. Otherwise we have sync problem.
    if len(left_images) != len(right_images):
        print("Numbers of left and right images are not equal. They should be pairs.")
        print("Left images count: ", len(left_images))
        print("Right images count: ", len(right_images))
        sys.exit(-1)

    pair_images = zip(left_images, right_images)  # Pair the images for single loop handling

    # Iterate through the pairs and find chessboard corners. Add them to arrays
    # If openCV can't find the corners in one image, we discard the pair.
    for left_im, right_im in pair_images:
        # Right Object Points
        right = cv2.imread(right_im)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size,
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        # Left Object Points
        left = cv2.imread(left_im)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size,
                                                           cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        if ret_left and ret_right:
            # Object points
            objpoints.append(objp)
            # Right points
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (5, 5), (-1, -1), criteria)
            imgpointsR.append(corners2_right)
            # Left points
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (5, 5), (-1, -1), criteria)
            imgpointsL.append(corners2_left)
        else:
            print("Chessboard couldn't detected. Image pair: ", left_im, " and ", right_im)
            continue

    image_size = gray_right.shape  # If you have no acceptable pair, you may have an error here.
    return [objpoints, imgpointsL, imgpointsR]

if __name__ == '__main__':
    # Check the help parameters to understand arguments
    parser = argparse.ArgumentParser(description='Stereo Calibration')
    parser.add_argument('--paramL', type=str, required=False, default='./stereoCalibParamL.yml', help='left matrix YML file')
    parser.add_argument('--paramR', type=str, required=False, default='./stereoCalibParamR.yml', help='right matrix YML file')
    parser.add_argument('--prefixL', type=str, required=False, default= '', help='left image prefix')
    parser.add_argument('--prefixR', type=str, required=False, default= '', help='right image prefix')
    parser.add_argument('--dirL', type=str, required=False, default='CalibDataStereo/left', help='left images directory path')
    parser.add_argument('--dirR', type=str, required=False, default='CalibDataStereo/right', help='right images directory path')
    parser.add_argument('--imageFormat', type=str, required=False, default='png', help='image format, png/jpg')
    parser.add_argument('--width', type=int, required=False, default=8, help='chessboard width size, default is 8')
    parser.add_argument('--height', type=int, required=False, default=5, help='chessboard height size, default is 5')
    parser.add_argument('--squareSize', type=float, required=False, default=27.0, help='chessboard square size')
    parser.add_argument('--saveFile', type=str, required=False, default='stereoCalibParam.yml', help='YML file to save stereo calibration matrices')

    args = parser.parse_args()
    # If chessboard pattern is different, we will pass them as arguments.
    if args.width is None and args.height is None:
        stereoCalibrate(args.paramL, args.paramR, args.dirL, args.dirR, args.prefixR, args.imageFormat, args.saveFile, args.squareSize)
    else:
        stereoCalibrate(args.paramL, args.paramR, args.dirL, args.prefixL, args.dirR, args.prefixR, args.imageFormat, args.saveFile, args.squareSize, args.width, args.height)