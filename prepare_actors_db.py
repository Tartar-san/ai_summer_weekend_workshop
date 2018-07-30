import os
import numpy as np
import cv2
from tqdm import tqdm
from recognition_model import get_embeddings

dataset_path = "imdb_extracted"

# TODO: Recompute embeddings for actors using MTCNN and save them in efficient way

for celebrity in os.listdir(dataset_path):
    print(celebrity)
    for filename in tqdm(os.listdir(os.path.join(dataset_path, celebrity))):
        if filename[-3:] == "jpg":
            # print(filename)
            try:
                filepath = os.path.join(os.path.join(dataset_path, celebrity), filename)
                img = cv2.imread(filepath)
                faces, embeddings = get_embeddings(img, True)
                if embeddings is None:
                    os.remove(filepath)
                embeddings_path = os.path.join(os.path.join(dataset_path, celebrity), os.path.basename(filename)[:-4]+".npy")
                np.save(embeddings_path, embeddings)
            except:
                os.remove(filepath)
