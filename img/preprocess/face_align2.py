'''
Original author: XY Feng
Edited by: Ishank Sharma
'''



import dlib
import numpy as np
import os
import glob
import cv2

FOLDER_PATH = "/normal"
IMAGE_FORMAT = "jpg"
REFERENCE_PATH = "resz0.jpg"
SCALE_FACTOR = 1
PREDICTOR_PATH = "face_landmarks.dat"



#FACE_POINTS = list(range(17, 68))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
#JAW_POINTS = list(range(0, 17))
#MOUTH_POINTS = list(range(48, 61))
#RIGHT_BROW_POINTS = list(range(17, 22))
#LEFT_BROW_POINTS = list(range(22, 27))


ALIGN_POINTS = (RIGHT_EYE_POINTS + LEFT_EYE_POINTS + NOSE_POINTS)

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im,fname):
    rects = detector(im, 1)
    maxbb=max(rects, key=lambda rect: rect.width() * rect.height())
    return np.matrix([[p.x, p.y] for p in predictor(im, maxbb).parts()])

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)

    try:
        s = get_landmarks(im,fname)
        return im, s
    except:    
        return im,fname
    
def nobox(img):
    cv2.imwrite('g_nor_nobox_'+ str(img_index)+'.jpg', img)

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,M[:2],(dshape[1], dshape[0]),dst=output_im,borderMode=cv2.BORDER_TRANSPARENT,flags=cv2.WARP_INVERSE_MAP)
    return output_im

img_ref, landmark_ref = read_im_and_landmarks(REFERENCE_PATH)
img_index = 0

for f in glob.glob(os.path.join(FOLDER_PATH, "*." + IMAGE_FORMAT)):
    print("Processing file: {}".format(f))
    img_index += 1
    img, landmark = read_im_and_landmarks(f)
    try:
        M = transformation_from_points(landmark_ref[ALIGN_POINTS],landmark[ALIGN_POINTS])
        warped_im2 = warp_im(img, M, img_ref.shape)
        cv2.imwrite('g_nor_align_'+str(img_index)+'.jpg', warped_im2)
    except:
        nobox(img)
        
   