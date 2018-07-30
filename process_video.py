import os
import numpy as np
import cv2
from recognition_model import get_embeddings, compare_embeddings
from face_detector import detect_faces

video_to_process = "testing_video/johny.mp4"
actors_dataset_path = "imdb_extracted"


def load_actors_embeddings(dataset_path):
    """ Loads precomputed embeddings and returns a list with actors and np array with embeddings"""
    embeddings = []
    actors = []
    for celebrity in os.listdir(dataset_path):
        cel_path = os.path.join(dataset_path, celebrity)
        for filename in os.listdir(cel_path):
            if filename[-3:] == "npy":
                embedding = np.load(os.path.join(cel_path, filename))
                actors.append(celebrity)
                embeddings.append(embedding)
    embeddings = np.array(embeddings)
    return actors, embeddings


actors, actor_embeddings = load_actors_embeddings(actors_dataset_path)

video = cv2.VideoCapture(video_to_process)
fps = video.get(cv2.CAP_PROP_FPS)

ret = True
frame_counter = 0

# Detections are saved in format {name : [[start_time, end_time]]}
detections = {}
working_memory = {}
thresh = float("inf")

while(ret):
    # Capture frame-by-frame
    ret, frame = video.read()
    frame_counter += 1

    if ret:
        if frame_counter % 25 == 0:
            matches = []
            faces = detect_faces(frame)
            embeddings_from_video = get_embeddings(frame, faces)
            for video_embd in embeddings_from_video:
                similiarity = compare_embeddings(video_embd, actor_embeddings)
                idx = np.argmax(similiarity)
                if similiarity[idx] < thresh:
                    matches.append(actors[idx])

            for match in matches:
                if match not in working_memory.keys():
                    working_memory[match] = frame_counter / fps

            matched_earlier = list(working_memory.keys())

            for matched in matched_earlier:
                if matched not in matches:
                    detections[matched] = [working_memory[matched], frame_counter / fps]
                    del working_memory[matched]

print(detections)








