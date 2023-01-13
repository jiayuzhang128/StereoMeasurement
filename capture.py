#  ==================================================================================
#  作   者：张佳预
#  日   期：2023/1/13
#  版   本：v1.0
#  代码描述：用于获取单目相机或双目相机的标定图片
#  软件平台：python==3.7.5
#  运   行:
#      直接点击即可运行
#  ==================================================================================

import cv2
import os


def makeDir(path='ClibData', numcam=1):
    '''
    创建用于存放标定图片的文件夹\n
    参数：\n
        path：文件夹名称，默认为‘ClibData’\n
        numcam：相机数量，默认为1
    '''
    if numcam == 1:
        path += 'Mono'
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
            print(path, ' 文件夹创建成功')
        else:
            print(path, ' 文件夹已经存在')
        return path
    elif numcam == 2:
        path += 'Stereo'
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
            os.makedirs(path + r'/left')
            os.makedirs(path + r'/right')
            print(path, ' 文件夹创建成功')
            print(path + r'/left', ' 文件夹创建成功')
            print(path + r'/right', ' 文件夹创建成功')
            return path
        else:
            print(path, ' 文件夹已经存在')
        return path
    else:
        print("文件夹创建失败！请将摄像头数量numcam设置为1或2！")
        return 0


def getPicture1(index, savePath, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001), numCorner1=8, numCorner2=5, idImage=0):
    '''
    拍摄单目相机标定图像\n
    参数：\n
    index：相机索引\n
    savePath：图片保存路径\n
    criteria：获得亚像素角点算法终止条件，默认为‘最大迭代30次或误差小于0.001时终止’\n
    numCorner1：棋盘格行角点数，默认为8\n
    numCorner2：棋盘格行角点数，默认为5\n
    idImage：保存图片的序号
    '''
    # 调用摄像头
    cam = cv2.VideoCapture(index)   # index -> 摄像头索引

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, (numCorner1, numCorner2), None)

        if ret == True:
            # 获得亚像素角点坐标
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            # 绘制并显示角点
            cv2.drawChessboardCorners(
                gray, (numCorner1, numCorner2), corners2, ret)
        cv2.imshow('Video', gray)

        key = cv2.waitKey(1)
        # 按‘s’健保存图片
        if (key & 0xFF == ord('s')):
            if ret == True:
                strIdImage = str(idImage)
                cv2.imwrite(savePath + '/cam' + strIdImage + '.png', frame)
                print('第{}张图片，保存成功'.format(idImage))
                idImage = idImage+1
            else:
                print('保存失败！棋盘格不完整，请换个角度重新保存！')

        # 按'ESC'退出程序
        if key & 0xFF == 27:
            print('程序已终止！一共保存了{}张图片'.format(idImage))
            break

    # 释放摄像头
    cam.release()
    cv2.destroyAllWindows()


def getPicture2(index1, index2, savePath, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001), numCorner1=8, numCorner2=5, idImage=0):
    '''
    拍摄双目相机标定图像\n
    参数：\n
    index1：左相机索引\n
    index2：右相机索引\n
    savePath：图片保存路径\n
    criteria：获得亚像素角点算法终止条件，默认为‘最大迭代30次或误差小于0.001时终止’\n
    numCorner1：棋盘格行角点数，默认为8\n
    numCorner2：棋盘格行角点数，默认为5\n
    idImage：保存图片的序号
    '''

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 调用摄像头
    CamR = cv2.VideoCapture(index1)   # index1 -> Right Camera
    CamL = cv2.VideoCapture(index2)   # index2 -> Left Camera

    while True:
        retR, frameR = CamR.read()
        retL, frameL = CamL.read()

        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retR, cornersR = cv2.findChessboardCorners(
            grayR, (numCorner1, numCorner2), None)
        retL, cornersL = cv2.findChessboardCorners(
            grayL, (numCorner1, numCorner2), None)

        if (retR == True) & (retL == True):
            # 获得亚像素角点坐标
            corners2R = cv2.cornerSubPix(
                grayR, cornersR, (11, 11), (-1, -1), criteria)
            corners2L = cv2.cornerSubPix(
                grayL, cornersL, (11, 11), (-1, -1), criteria)

            # 绘制并显示角点
            cv2.drawChessboardCorners(
                grayR, (numCorner1, numCorner2), corners2R, retR)
            cv2.drawChessboardCorners(
                grayL, (numCorner1, numCorner2), corners2L, retL)
        cv2.imshow('VideoR', grayR)
        cv2.imshow('VideoL', grayL)

        key = cv2.waitKey(1)
        # 按‘s’健保存图片
        if (key & 0xFF == ord('s')):
            if (retR == True) & (retL == True):
                strIdImage = str(idImage)
                cv2.imwrite(savePath + '/right/right' +
                            strIdImage + '.png', frameR)
                cv2.imwrite(savePath + '/left/left' +
                            strIdImage + '.png', frameL)
                print('第{}张图片，保存成功'.format(idImage))
                idImage = idImage+1
            else:
                print('保存失败！棋盘格不完整，请换个角度重新保存！')

        # 按'ESC'退出程序
        if key & 0xFF == 27:
            print('程序已终止！一共保存了{}张图片'.format(idImage))
            break

    # 释放相机
    CamR.release()
    CamL.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    print('Starting the Calibration. Press and maintain the ESC key to exit the script\n')
    print('Push (s) to save the image')
    path = makeDir()
    if path:
        getPicture1(index=0, savePath=path)
