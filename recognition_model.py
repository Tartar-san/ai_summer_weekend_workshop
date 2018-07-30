import net_sphere
import cv2
import torch
import numpy as np
import imutils
from utils import crop_bb
from torch.autograd import Variable
from face_detector import detect_faces
from matlab_cp2tform import get_similarity_transform_for_cv2

torch.set_num_threads(4)

MODEL = 'sphere20a'
MODEL_PATH = 'sphere20a.pth'


def alignment(src_img, src_pts):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    crop_size = (96, 112)

    src_pts = [src_pts["keypoints"]["left_eye"], src_pts["keypoints"]["right_eye"], src_pts["keypoints"]["nose"],
               src_pts["keypoints"]["mouth_left"], src_pts["keypoints"]["mouth_right"]]
    src_pts = np.array(src_pts)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def get_embeddings(img, faces):
    """ Return embeddings of the faces in image using spherenet model """

    processed_imgs = []

    for pts in faces:
        face_alignment = alignment(img, pts)
        processed_img = face_alignment.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        processed_img = (processed_img - 127.5) / 128.0
        processed_imgs.append(processed_img)

    processed_img = np.vstack(processed_imgs)
    processed_img = Variable(torch.from_numpy(processed_img).float(), volatile=True)
    output = net(processed_img)
    embeddings = output.data.numpy()

    return embeddings


def compare_embeddings(embedding, embeddings):
    embedding_norm = np.linalg.norm(embedding)
    divider = (np.linalg.norm(embeddings, axis=1) * embedding_norm + 1e-5)[:, None]
    cosdistances = np.dot(embeddings / divider, embedding)
    return cosdistances


net = getattr(net_sphere, 'sphere20a')()
net.load_state_dict(torch.load('sphere20a.pth'))
net.eval()
net.feature = True

if __name__ == "__main__":

    import os

    emebeddings = []

    for filename in os.listdir("testing_images"):
        image = cv2.imread("testing_images/" + filename)
        faces = detect_faces(image)
        emebeddings.append(get_embeddings(image, faces))

    print(emebeddings)


