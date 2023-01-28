import numpy as np
import cv2
import glob
import argparse
from calibrationStore import saveCoefficients

# 终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def monoCalib(dirPath, prefix, imageFormat, saveFile, squareSize, width=8, height=5):
    """ 单目标定函数 """
    # 生成真实角点相对坐标 (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    # 生成真实角点绝对坐标
    objp = objp * squareSize

    # 储存所有图像中真实角点坐标和图像角点坐标
    objpoints = []  # 世界坐标系3d点
    imgpoints = []  # 图像坐标系2d点

    # 路径校正，确保最后一个字符不是'/'
    if dirPath[-1:] == '/':
        dirPath = dirPath[:-1]

    # 获得图片
    images = glob.glob(dirPath+'/' + prefix + '*.' + imageFormat)
    # print(images)

    # 迭代遍历所有图像，将每张图像角点的真实坐标和图像坐标添加到数组中
    for name in images:
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # 绘制并显示棋盘格角点
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            # cv2.imshow(img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    saveCoefficients(saveFile, mtx, dist, ret)

    print("Calibration is finished. RMS: ", ret)

    return [ret, mtx, dist, rvecs, tvecs]


if __name__ == '__main__':
    # 使用'-h'查看各个参数的帮助
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--imageDir', type=str, required=False, default='CalibDataMono', help='image directory path')
    parser.add_argument('--imageFormat', type=str, required=False, default='png', help='image format, png/jpg')
    parser.add_argument('--prefix', type=str, required=False, default='', help='image prefix')
    parser.add_argument('--squareSize', type=float, required=False, default=27.0, help='chessboard square size')
    parser.add_argument('--width', type=int, required=False, default=8, help='chessboard width size, default is 8')
    parser.add_argument('--height', type=int, required=False, default=5, help='chessboard height size, default is 5')
    parser.add_argument('--saveFile', type=str, required=False, default='monoCalibParam.yml', help='YML file to save calibration matrices')

    args = parser.parse_args()

    # 调用标定函数. RMS是误差, 误差小于0.2为佳
    ret, mtx, dist, rvecs, tvecs = monoCalib(args.imageDir, args.prefix, args.imageFormat, args.saveFile,args.squareSize, args.width, args.height)
