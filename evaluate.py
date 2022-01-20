import cv2
import json
import numpy as np
import argparse
import torch

from model.skel import skel_net
from utils.json_parse import parse_openpose
from train import parse_config

IMAGE_WIDTH = 1080

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
        default='./configs/default.json',
        type=str, help='config file (default: default.json)'
    )

    args = parser.parse_args()
    if args.config:
        config = parse_config(args)
    else:
        AssertionError("missing config file path.")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    file_name = './data/0/0.json'
    pose_batch = []
    confidence_batch = []
    with open(file_name, 'r') as f:
        pose, confidence = parse_openpose(json.load(f))
        pose[15] = pose[22]
        pose[16] = pose[19]
        pose = pose[:17]
        confidence[15] = confidence[22]
        confidence[16] = confidence[19]
        confidence = confidence[:17]
        pose_batch.append(pose)
        confidence_batch.append(confidence)

    # img = np.ones((IMAGE_WIDTH, IMAGE_WIDTH, 3), np.uint8)
    # for loc, conf in zip(pose_batch[0], confidence_batch[0]):
    #     cv2.circle(img, (int(loc[0]), int(loc[1])), 1, color=(0, 0, 255 * conf), thickness=-1)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    poses_2d = np.concatenate(pose_batch, axis=0)/IMAGE_WIDTH
    poses_2d = np.reshape(poses_2d, (-1, 34))
    # poses_2d_root = (poses_2d - np.tile(poses_2d[:, :2], [1, int(poses_2d.shape[-1] / 2)]))

    # poses_2d_root = np.where(np.isfinite(poses_2d_root), poses_2d_root, 0)
    poses_2d_root = torch.from_numpy(np.array(poses_2d)).unsqueeze(0).float().to(device)
    skel = skel_net(config)
    skel.forward(poses_2d_root)
    print(skel)

    # run model
    # model.forward()
    # result = bone scale & bone rotations
    # project skeleton to camera.
    # show image

if __name__ == '__main__':
    main()