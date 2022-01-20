import numpy as np

def parse_openpose(openpose_json):
    people = openpose_json['people']
    if len(people) != 1:
        return None

    keypoints = np.array(openpose_json['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

    locations = []
    confidences = []
    for i in range(len(keypoints)):
        locations.append(keypoints[i][:2])
        confidences.append(keypoints[i][2])

    return np.array(locations), np.array(confidences)