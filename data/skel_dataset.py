import torch
import os, json
import numpy as np

from utils.json_parse import parse_openpose

openpose_to_myskel = [8, 9, 10, 11, 22, 12, 13, 14, 19, 1, 0, 5, 6, 7, 2, 3, 4]
#0 Hips
#1 RightUpLeg
#2 RightLeg
#3 RightFoot
#4 RightBigToe
#5 LeftUpLeg
#6 LeftLeg
#7 LeftFoot
#8 LeftBigToe
#9 Spine
#10 Spine1
#11 Neck
#12 Head
#13 LeftArm
#14 LeftForeArm
#15 LeftHand
#16 RightArm
#17 RightForeArm
#18 RightHand

class SkelDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.pose_data = []
        input_dir = os.path.join(os.path.dirname(__file__), 'input')
        files = os.listdir(input_dir)
        for file in files:
            filepath = os.path.join(input_dir, file)
            with open(filepath, 'r') as f:
                js = json.load(f)
                result = parse_openpose(js)
                if result is None:
                    continue
                
                pose, confidence = result
                # pose[15] = pose[22]
                # pose[16] = pose[19]

                real_pose = np.zeros((17,2), dtype=np.float32)
                for i in range(len(openpose_to_myskel)):
                    real_pose[i] = pose[openpose_to_myskel[i]]

                # confidence[15] = confidence[22]
                # confidence[16] = confidence[19]
                real_confidence = np.zeros((17,2), dtype=np.float32)
                for i in range(len(openpose_to_myskel)):
                    real_confidence[i] = confidence[openpose_to_myskel[i]]

                no_conf = 0
                for c in range(len(real_confidence)):
                    if real_confidence[c][0] == 0 or real_confidence[c][1] == 0:
                        no_conf = 1

                if no_conf:
                    continue
                        
                self.pose_data.append(np.reshape(real_pose, (34)))
                # confidence_batch.append(confidence)

        print(len(self.pose_data))

    def __len__(self):
        return len(self.pose_data)

    def __getitem__(self, idx):
        return self.pose_data[idx], self.pose_data[idx]