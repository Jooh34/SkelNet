import os
from utils.bvh import Bvh
import torch
from queue import Queue
import numpy as np

import matplotlib.pyplot as plt

my_skel = [
    'Hips',
    'RightUpLeg',
    'RightLeg',
    'RightFoot',
    'RightBigToe',
    'LeftUpLeg',
    'LeftLeg',
    'LeftFoot',
    'LeftBigToe',
    'Spine',
    'Spine1',
    'Neck',
    'Head',
    'LeftArm',
    'LeftForeArm',
    'LeftHand',
    'RightArm',
    'RightForeArm',
    'RightHand',
]
my_skel_offset_list = []
my_skel_bone_hierarcy = []

openpose_to_myskel = [12, 11, 16, 17, 18, 13, 14, 15, 0, 1, 2, 3, 5, 6, 7, 4, 8]
# openpose_to_myskel = [
#     'Head',
#     'Neck',
#     'RightArm',
#     'RightForeArm',
#     'RightHand',
#     'LeftArm',
#     'LeftForeArm',
#     'LeftHand'
#     'Hips',
#     'RightUpLeg',
#     'RightLeg',
#     'RightFoot',
#     'LeftUpLeg',
#     'LeftLeg',
#     'LeftFoot',
#     'RightBigToe',
#     'LeftBigToe',
# ]

class BoneNode:
    def __init__(self):
        self.parent = None
        self.children = []

    def set_parent_children(self, parent, children):
        self.parent = parent
        self.children = children


def setup_bone_hierarcy():
    for i in range(len(my_skel)):
        my_skel_bone_hierarcy.append(BoneNode())
    
    #Hips
    my_skel_bone_hierarcy[0].set_parent_children(None, [1,5,9])

    #Right Lower
    my_skel_bone_hierarcy[1].set_parent_children(0, [2])
    my_skel_bone_hierarcy[2].set_parent_children(1, [3])
    my_skel_bone_hierarcy[3].set_parent_children(2, [4])
    my_skel_bone_hierarcy[4].set_parent_children(3, [])

    #Left Lower
    my_skel_bone_hierarcy[5].set_parent_children(0, [6])
    my_skel_bone_hierarcy[6].set_parent_children(5, [7])
    my_skel_bone_hierarcy[7].set_parent_children(6, [8])
    my_skel_bone_hierarcy[8].set_parent_children(7, [])

    #Spine to Head
    my_skel_bone_hierarcy[9].set_parent_children(0, [10])

    my_skel_bone_hierarcy[10].set_parent_children(9, [11, 13, 16])
    my_skel_bone_hierarcy[11].set_parent_children(10, [12])
    my_skel_bone_hierarcy[12].set_parent_children(11, [])

    #Left Upper
    my_skel_bone_hierarcy[13].set_parent_children(10, [14])
    my_skel_bone_hierarcy[14].set_parent_children(13, [15])
    my_skel_bone_hierarcy[15].set_parent_children(14, [])

    #Right Upper
    my_skel_bone_hierarcy[16].set_parent_children(10, [17])
    my_skel_bone_hierarcy[17].set_parent_children(16, [18])
    my_skel_bone_hierarcy[18].set_parent_children(17, [])

def set_up():
    bvh_path = os.path.join(os.path.dirname(__file__), '../data/target.bvh')
    with open(bvh_path) as f:
        mocap = Bvh(f.read())
        for bone_name in my_skel:
            offset = mocap.joint_offset(bone_name)
            t = torch.tensor([
                [1, 0, 0, offset[0]],
                [0, 1, 0, offset[1]],
                [0, 0, 1, offset[2]],
                [0, 0, 0, 1],
            ], dtype=torch.float32).cuda()

            my_skel_offset_list.append(t)

    setup_bone_hierarcy()

def show_skeleton(fake_3d_position, title):
    # (num_joints, 3)
    points_list = fake_3d_position.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # do dfs
    q = Queue()
    q.put(0)

    while not q.empty():
        idx = q.get()
        node = my_skel_bone_hierarcy[idx]
        parent = node.parent

        p1 = points_list[idx]
        ax.text(p1[0], p1[1], p1[2], str(idx), size=10)

        if parent is not None:
            p2 = points_list[parent]
            ax.plot([p1[0] ,p2[0]],[p1[1],p2[1]], zs=[p1[2],p2[2]])

        for c in node.children:
            q.put(c)

    ax.set_title(title)
    plt.show()
    return

def show_difference_2d(Y, fake_Y):
    Y_np = Y.detach().cpu().numpy()
    fake_Y_np = fake_Y.detach().cpu().numpy()

    Y_np = Y_np.reshape((17,2))
    fake_Y_np = fake_Y_np.reshape((17,2))

    interp =  Y_np[0] + (Y_np[9]-Y_np[0])/2
    a = np.concatenate((Y_np[:9, :], np.array([interp, interp])), axis=0)
    Y_np = np.concatenate((a, Y_np[9:,:]), axis=0)

    interp =  fake_Y_np[0] + (fake_Y_np[9]-fake_Y_np[0])/2
    a = np.concatenate((fake_Y_np[:9, :], np.array([interp, interp])), axis=0)
    fake_Y_np = np.concatenate((a, fake_Y_np[9:,:]), axis=0)

    fig=plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax1 = fig.add_subplot(2, 2, 3)
    ax2 = fig.add_subplot(2, 2, 4)
    ax1.set_title('Y')
    ax2.set_title('fake_Y')

    # do dfs for Y
    q = Queue()
    q.put(0)

    while not q.empty():
        idx = q.get()
        node = my_skel_bone_hierarcy[idx]
        parent = node.parent

        p1 = Y_np[idx]
        ax1.text(p1[0], p1[1], str(idx), size=10)

        if parent is not None:
            p2 = Y_np[parent]
            ax1.plot([p1[0] ,p2[0]],[p1[1],p2[1]])

        for c in node.children:
            q.put(c)

    
    # do dfs for fake_Y
    q = Queue()
    q.put(0)

    while not q.empty():
        idx = q.get()
        node = my_skel_bone_hierarcy[idx]
        parent = node.parent

        p1 = fake_Y_np[idx]

        if parent is not None:
            p2 = fake_Y_np[parent]
            ax2.plot([p1[0] ,p2[0]],[p1[1],p2[1]])

        for c in node.children:
            q.put(c)

    plt.show()