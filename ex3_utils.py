import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 212733257


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    Gr_kernal = np.asarray([[1, 0, -1]])
    Ix = cv2.filter2D(im1, -1, Gr_kernal)
    Iy = cv2.filter2D(im1, -1, Gr_kernal.T)
    It = (im2 - im1)
    points = []
    u_v = []
    for i in range(win_size // 2, im1.shape[0], step_size):
        for j in range(win_size // 2, im1.shape[1], step_size):
            vecIx = Ix[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2:j + win_size // 2 + 1].flatten()
            vecIy = Iy[i - win_size // 2: i + win_size // 2 + 1, j - win_size // 2:j + win_size // 2 + 1].flatten()
            A = np.stack((vecIx, vecIy), axis=-1)
            At = A.T
            mat = At @ A
            B = It[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2:j + win_size // 2 + 1].flatten()
            AtB = At @ B
            if j == 152 and i == 62:
                print("stop")
            if np.linalg.det(mat) != 0:
                uv = np.linalg.inv(mat) @ AtB
                l1, l2 = np.linalg.eigvals(mat)
                if l1 >= l2 >= 0 and l1 / l2 <= 100:
                    points.append([j, i])
                    u_v.append([uv[0], uv[1]])
    return np.asarray(points), np.asarray(u_v)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    pyramid_img1 = []
    pyramid_img2 = []
    pyramid_img1.append(img1)
    pyramid_img2.append(img2)
    for i in range(1, k):
        Itemp = cv2.GaussianBlur(pyramid_img1[-1], (5,5), 0)
        Itemp = Itemp[::2, ::2]
        pyramid_img1.append(Itemp)

        Itemp = cv2.GaussianBlur(pyramid_img2[-1], (5,5), 0)
        Itemp = Itemp[::2, ::2]
        pyramid_img2.append(Itemp)
    pyramid_img1 = np.flip(pyramid_img1).tolist()
    pyramid_img2 = np.flip(pyramid_img2).tolist()

    # iterative method
    UV = list()
    for i in range(k-1):
        points, uv = opticalFlow(pyramid_img1[i], pyramid_img2[i], stepSize, winSize)
        UV = uv
        print(i)
        toWarp = pyramid_img2[i+1]
        movment = zip(points, uv)
        for p, d in movment:
            secondShift = int(p[0] * 2 + np.floor(d[0] * 2))
            firstShift = int(p[1] * 2 + np.floor(d[1] * 2))
            toWarp[p[1] * 2, p[0] * 2] = pyramid_img1[i+1][firstShift, secondShift]
    U = np.take(UV, axis=0)
    V = np.take(UV, axis=1)
    return np.stack((U, V, 2))


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    points, uv = opticalFlow(im1, im2)
    sum = np.sum(uv, axis=0)
    u = sum[0].mean()
    v = sum[1].mean()
    mat = np.zeroes((3, 3))
    mat[0, 0] = 1
    mat[1, 1] = 1
    mat[2, 2] = 1
    mat[0, 2] = u
    mat[1, 2] = v
    return mat


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pyr_list = list()
    pyr_list.append(img)
    for i in range(1, levels):
        sigma = 0.3 + 0.8
        Itemp = cv2.GaussianBlur(pyr_list[-1], (5, 5), sigma)
        Itemp = Itemp[::2, ::2]
        pyr_list.append(Itemp)
    return pyr_list


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pyrList = gaussianPyr(img, levels)
    lap_list = list()

    lap_list.append(pyrList[-1])

    rangeLevels = np.flip(np.arange(1, levels))

    for i in rangeLevels:
        expended = cv2.resize(pyrList[i], (pyrList[i].shape[1] * 2, pyrList[i].shape[0] * 2))
        if expended.shape != pyrList[i - 1].shape:
            expended = cv2.resize(expended, (pyrList[i - 1].shape[1], pyrList[i - 1].shape[0]))
        lap_list.append(np.asarray(pyrList[i - 1] - expended))

    return np.flip(lap_list).tolist()


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    lap_pyr = np.flip(lap_pyr)
    pyr_reversed = list()
    for i in range(len(lap_pyr)):
        if i == 0:
            pyr_reversed.append(lap_pyr[i])
        else:
            expended = cv2.resize(pyr_reversed[-1], (pyr_reversed[-1].shape[1] * 2, pyr_reversed[-1].shape[0] * 2))
            original = lap_pyr[i] + expended
            pyr_reversed.append(original)

    return pyr_reversed[-1]


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    lap1 = np.flip(laplaceianReduce(img_1, levels)).tolist()
    lap2 = np.flip(laplaceianReduce(img_2, levels)).tolist()
    mask_pyr = np.flip(gaussianPyr(mask, levels)).tolist()
    merge = list()
    merge.append(lap1[0] * mask_pyr[0] + (1 - mask_pyr[0]) * lap2[0])
    for i in range(1, levels):
        resized = cv2.resize(merge[-1], (lap1[i].shape[1], lap1[i].shape[0]))
        merge.append(resized + lap1[i] * mask_pyr[i] + (1 - mask_pyr[i]) * lap2[i])
    return np.zeros(merge[-1].shape), merge[-1]
