import cv2
import argparse
from calibrationStore import loadStereoCoefficients, saveStereoCoefficients, loadStereoImages
from capture import makeDir

def rectify(dirL, dirR, imageFormat, loadCalibFile, saveCalibFile):

    # 立体校正
    imageSize, K1, D1, K2, D2, R, T, E, F, rms = loadStereoCoefficients(loadCalibFile, rectifid=False)

    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(K1, D1, K2, D2, imageSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=imageSize)

    # 校正映射
    mapLx, mapLy = cv2.initUndistortRectifyMap(K1, D1, RL, PL, imageSize, cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(K2, D2, RR, PR, imageSize, cv2.CV_32FC1)

    saveStereoCoefficients(saveCalibFile,  imageSize, K1, D1, K2, D2, R, T, E, F, rms,RL, RR, PL, PR, Q, roiL, roiR, mapLx, mapLy, mapRx, mapRy)

    count = 0
    saveImagesPath = makeDir(numCam=2, path='RectifyData')

    print("开始图片校正，按‘c’继续，按‘ESC'退出\n")
    unrectifyImagesLR = loadStereoImages(dirL,dirR,imageFormat)

    for imageL,imageR in unrectifyImagesLR:
        unrectifyL = cv2.imread(imageL)
        unrectifyR = cv2.imread(imageR)

        rectifyL = cv2.remap(unrectifyL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectifyR = cv2.remap(unrectifyR, mapRx, mapRy, cv2.INTER_LINEAR)

        # 保存结果
        cv2.imwrite(saveImagesPath + "/left/" + str(count) + ".png", rectifyL)
        cv2.imwrite(saveImagesPath + "/right/" + str(count) + ".png", rectifyR)
    
        # 显示结果
        showRectify(unrectifyL,unrectifyR,rectifyL,rectifyR)

        # 输出提示信息
        count += 1
        print("图片对：" + imageL + " " + imageR + " 校正成功")
        print("已校正成功校正 " + str(count) + " 对图片\n")

        key = cv2.waitKey(0)
        if key == 'c':
            continue
        elif key == 27:
            break

def showRectify(unrectifyL,unrectifyR,rectifyL,rectifyR):

    unrectifyLR = cv2.hconcat([unrectifyL, unrectifyR])
    rectifyLR = cv2.hconcat([rectifyL, rectifyR])

    cv2.rectangle(rectifyLR, (0,0), (1280,720),(255,0,0),3)
    cv2.rectangle(rectifyLR, (1280,0), (2560,720),(0,0,255),3)

    mixLR = cv2.vconcat([unrectifyLR,rectifyLR])

    h, w, _ = mixLR.shape

    scalar = 1200/max(h, w) # 缩放图像宽度为1200，方便可视化
    H = int(h * scalar)
    W = int(w * scalar)
    scaleMixLR = cv2.resize(mixLR, (W,H), interpolation=cv2.INTER_LINEAR)

    for i in range(0, H,16):
        cv2.line(scaleMixLR, (0,i), (W,i), (0,255,0))

    cv2.namedWindow("result",cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result",scaleMixLR)


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Stereo Rectify')
    parser.add_argument('--dirL', type=str, required=False, default='CalibDataStereo/left', help='left unrectified images directory path')
    parser.add_argument('--dirR', type=str, required=False, default='CalibDataStereo/right', help='right unrectified images directory path')
    parser.add_argument('--imageFormat', type=str, required=False, default='png', help='image format, png/jpg')
    parser.add_argument('--loadCalibFile', type=str, required=False, default='./stereoCalibParam.yml', help='name of stereo calibration data YML file')
    parser.add_argument('--saveCalibFile', type=str, required=False, default='./RectifyStereoCalibParam.yml', help='name of rectified YML file')

    args = parser.parse_args()

    rectify(args.dirL, args.dirR, args.imageFormat, args.loadCalibFile, args.saveCalibFile)