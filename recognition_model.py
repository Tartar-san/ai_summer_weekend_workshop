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


def alignment(src_img, bb):
    """ Crop faces and preprocess them for spherenet """
    face = crop_bb(src_img, bb)
    crop_size = (96, 112)
    face_resized = cv2.resize(face, crop_size, interpolation=cv2.INTER_AREA)
    return face_resized


def get_embeddings(img, already_face=False):
    """ Return embeddings of the face image using spherenet model """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if already_face:
        bbxs = [[0, 0, gray.shape[0], gray.shape[1]]]
    else:
        bbxs = detect_faces(gray)

    if len(bbxs) == 0:
        return None, None

    faces = []
    processed_imgs = []

    for bb in bbxs:
        face = alignment(img, bb)
        faces.append(face)
        processed_img = face.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        processed_img = (processed_img - 127.5) / 128.0
        processed_imgs.append(processed_img)

    processed_img = np.vstack(processed_imgs)
    processed_img = Variable(torch.from_numpy(processed_img).float(), volatile=True)
    output = net(processed_img)
    embeddings = output.data.numpy()

    return faces, embeddings


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
        emebeddings.append(get_embeddings(image))

    print(emebeddings)


