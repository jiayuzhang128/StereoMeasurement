import cv2
import glob
import sys


def saveCoefficients(path, mtx, dist, rms):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.write("RMS", rms)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def saveStereoCoefficients(path, imageSize, K1, D1, K2, D2, R, T, E, F, rms, R1=None, R2=None, P1=None, P2=None, Q=None, roiL=None, roiR=None, mapLx=None, mapLy=None, mapRx=None, mapRy=None):
    """ Save the stereo coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("Size", imageSize)
    cv_file.write("K1", K1)
    cv_file.write("D1", D1)
    cv_file.write("K2", K2)
    cv_file.write("D2", D2)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.write("E", E)
    cv_file.write("F", F)
    cv_file.write("RMS", rms)
    if all(i is not None for i in [R1, R2, P1, P2, Q, roiL, roiR, mapLx, mapLy, mapRx, mapRy]):
        cv_file.write("R1", R1)
        cv_file.write("R2", R2)
        cv_file.write("P1", P1)
        cv_file.write("P2", P2)
        cv_file.write("Q", Q)
        cv_file.write("ROIL", roiL)
        cv_file.write("ROIR", roiR)
        cv_file.write("MAPLX", mapLx)
        cv_file.write("MAPLY", mapLy)
        cv_file.write("MAPRX", mapRx)
        cv_file.write("MAPRY", mapRy)
    cv_file.release()


def loadCoefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def loadStereoCoefficients(path, rectifid=False):
    """ Loads stereo matrix coefficients. """
    # 读取文件
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    imageSize = cv_file.getNode("Size").mat()
    Size = tuple([int(imageSize[1,0]),int(imageSize[0,0])])
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    rms = cv_file.getNode("RMS").real()
    if rectifid == True:
        R1 = cv_file.getNode("R1").mat()
        R2 = cv_file.getNode("R2").mat()
        P1 = cv_file.getNode("P1").mat()
        P2 = cv_file.getNode("P2").mat()
        Q = cv_file.getNode("Q").mat()
        roiL = cv_file.getNode("ROIL").mat()
        roiR = cv_file.getNode("ROIR").mat()
        mapLx = cv_file.getNode("MAPLX").mat()
        mapLy = cv_file.getNode("MAPLY").mat()
        mapRx = cv_file.getNode("MAPRX").mat()
        mapRy = cv_file.getNode("MAPRY").mat()
        result = [Size, K1, D1, K2, D2, R, T, E, F, rms, R1, R2, P1, P2, Q, roiL, roiR, mapLx, mapLy, mapRx, mapRy]
    else:
        result = [Size, K1, D1, K2, D2, R, T, E, F, rms]

    cv_file.release()
    return result

def loadStereoImages(dirL, dirR, imageFormat):

    # 图像路径校正. 移除'/'
    if dirL[-1:] == '/':
        dirL = dirL[:-1]
    if dirR[-1:] == '/':
        dirR = dirR[:-1]

    # 获取该路径下的所有图片
    imagesL = glob.glob(dirL + '/' + '*.' + imageFormat)
    imagesR = glob.glob(dirR + '/' + '*.' + imageFormat)
    imagesL.sort()
    imagesR.sort()

    # 确认找到图片
    if len(imagesL) & len(imagesR) == False:
        print("未找到图片，请检查路径后重新加载!")

    # 确保图片数量一致
    if len(imagesL) != len(imagesR):
        print("左图数量: ", len(imagesL))
        print("右图数量: ", len(imagesR))
        print("左右图像不均等，请检查路径后重新加载！")
        sys.exit(-1)
    
    imagesLR = zip(imagesL,imagesR)
    return imagesLR
