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


def makeDir(numCam=1, path='CalibData'):
    '''
    创建用于存放标定图片的文件夹\n
    参数：\n
        numCam：相机数量，默认为1\n
        path：文件夹名称，默认为‘ClibData’
    '''
    if numCam == 1:
        path += 'Mono'
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
            print(path, ' 文件夹创建成功')
        else:
            print(path, ' 文件夹已经存在')
        return path
    elif numCam == 2:
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
        print("文件夹创建失败！请将摄像头数量numCam设置为1或2！")
        return 0


def getPicture(numCam, index, savePath, width=640, height=480, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001), numCorner1=8, numCorner2=5, idImage=0):
    '''
    拍摄相机标定图像\n
    参数：\n
    numCam：相机数量
    index：相机索引\n
    savePath：图片保存路径\n
    width：图片宽
    height：图片高
    criteria：获得亚像素角点算法终止条件，默认为‘最大迭代30次或误差小于0.001时终止’\n
    numCorner1：棋盘格行角点数，默认为8\n
    numCorner2：棋盘格行角点数，默认为5\n
    idImage：保存图片的序号
    '''
    # 调用摄像头
    cap = cv2.VideoCapture(index)                # index -> 摄像头索引
    # 设置视频流属性，一般不要随意调整相机属性
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)     # 宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)   # 高度
    # cap.set(cv2.CAP_PROP_AUTO_WB, 1)             # 自动白平衡
    # cap.set(cv2.CAP_PROP_FPS, 30)                # 帧率30
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)          # 亮度0
    # cap.set(cv2.CAP_PROP_CONTRAST, 0)            # 对比度0
    # cap.set(cv2.CAP_PROP_SATURATION, 36)         # 饱和度36
    # cap.set(cv2.CAP_PROP_HUE, 0)                 # 色调0
    # cap.set(cv2.CAP_PROP_EXPOSURE, -6)           # 曝光-6


    if numCam == 2:
        # 定义显示窗口
        cv2.namedWindow("Video2",cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            frameL = frame[0:720,0:1280]
            frameR = frame[0:720,1280:2560]
            camL = frameL.copy()
            camR = frameR.copy()
            grayL = cv2.cvtColor(camL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(camR, cv2.COLOR_BGR2GRAY)

            # 寻找棋盘格角点
            retL, cornersL = cv2.findChessboardCorners(grayL, (numCorner1, numCorner2), None)
            retR, cornersR = cv2.findChessboardCorners(grayR, (numCorner1, numCorner2), None)

            if retL == True & retR == True:
                # 获得亚像素角点坐标
                corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
                corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
                # 绘制并显示角点
                cv2.drawChessboardCorners(camL, (numCorner1, numCorner2), corners2L, retL)
                cv2.drawChessboardCorners(camR, (numCorner1, numCorner2), corners2R, retR)
            # 两个框合并
            camLR = cv2.hconcat([camL,camR])
            cv2.imshow('Video2', camLR)
            # cv2.imshow('VideoL', camL)
            # cv2.imshow('VideoR', camR)

            key = cv2.waitKey(1)
            # 按‘s’健保存图片
            if (key & 0xFF == ord('s')):
                if retL == True & retL == True:
                    strIdImage = str(idImage)
                    cv2.imwrite(savePath + '/left/' + strIdImage + '.png', frameL)
                    cv2.imwrite(savePath + '/right/' + strIdImage + '.png', frameR)
                    print('第{}张图片，保存成功'.format(idImage))
                    idImage = idImage+1
                else:
                    print('保存失败！棋盘格不完整，请换个角度重新保存！')

            # 按'ESC'退出程序
            elif key & 0xFF == 27:
                print('程序已终止！一共保存了{}张图片'.format(idImage))
                break

    elif numCam == 1:
        cv2.namedWindow("Video1",cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            cam = frame.copy()
            gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
            # 寻找棋盘格角点
            rets, corners = cv2.findChessboardCorners(gray, (numCorner1, numCorner2), None)

            if rets == True:
                # 获得亚像素角点坐标
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # 绘制并显示角点
                cv2.drawChessboardCorners(
                    cam, (numCorner1, numCorner2), corners2, rets)
            cv2.imshow('Video1', cam)

            key = cv2.waitKey(1)
            # 按‘s’健保存图片
            if (key & 0xFF == ord('s')):
                if rets == True:
                    strIdImage = str(idImage)
                    cv2.imwrite(savePath + '/' + strIdImage + '.png', frame)
                    print('第{}张图片，保存成功'.format(idImage))
                    idImage = idImage+1
                else:
                    print('保存失败！棋盘格不完整，请换个角度重新保存！')

            # 按'ESC'退出程序
            if key & 0xFF == 27:
                print('程序已终止！一共保存了{}张图片'.format(idImage))
                break

    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    print('Starting the Calibration. Press and maintain the ESC key to exit the script\n')
    print('Push (s) to save the image')
    # path = makeDir(numCam=2)
    path = makeDir(numCam=1)
    if path:
        getPicture(numCam=1, index=1, savePath=path)
        # getPicture(numCam=2, index=0, savePath=path, width=2560,height=720)
