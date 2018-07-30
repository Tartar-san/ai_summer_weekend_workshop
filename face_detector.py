import cv2
from mtcnn.mtcnn import MTCNN
from utils import rect_to_bb, crop_bb

detector = MTCNN()


def detect_faces(img):
    """ Detect faces in gray opencv image and returns cropped faces """
    cropped_faces = []
    faces = detector.detect_faces(img)
    for face in faces:
        box = face["box"]
        cropped_face = crop_bb(img, box)
        cropped_faces.append(cropped_face)
    return cropped_faces

if __name__ == "__main__":

    import os
    for filename in os.listdir("testing_images"):
        image = cv2.imread("testing_images/"+filename)
        faces = detect_faces(image)
        for face in faces:
            cv2.imshow("1", face)
            cv2.waitKey(0)