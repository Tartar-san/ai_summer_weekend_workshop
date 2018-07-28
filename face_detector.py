import dlib
import cv2
from utils import rect_to_bb, crop_bb

detector = dlib.get_frontal_face_detector()


def detect_faces(gray_img):
    """ Detect faces in gray opencv image and returns cropped faces """
    rects = detector(gray_img, 1)
    return [rect_to_bb(r) for r in rects]

if __name__ == "__main__":

    import os
    for filename in os.listdir("testing_images"):
        image = cv2.imread("testing_images/"+filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bbxs = detect_faces(gray)
        for bb in bbxs:
            face = crop_bb(image, bb)
            cv2.imshow("1", face)
            cv2.waitKey(0)