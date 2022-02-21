import open3d as o3d
import open3d.visualization as vis
from queue import Queue

def create_scene(np_points, np_lines):
    points = o3d.utility.Vector3dVector(np_points)
    lines = o3d.utility.Vector2iVector(np_lines)

    line_set = o3d.geometry.LineSet(points, lines)
    geoms = [{
        "name": "skel",
        "geometry": line_set,
    }]

    return geoms

def draw_and_show_skeleton(fake_3d_position, bone_hierarcy):
    # (num_joints, 3)
    np_points = fake_3d_position.detach().cpu().numpy()
    np_lines = []    

    # do dfs
    q = Queue()
    q.put(0)

    while not q.empty():
        idx = q.get()
        node = bone_hierarcy[idx]
        parent = node.parent

        if parent is not None:
            np_lines.append([idx, parent])

        for c in node.children:
            q.put(c)

    points = o3d.utility.Vector3dVector(np_points)
    lines = o3d.utility.Vector2iVector(np_lines)

    line_set = o3d.geometry.LineSet(points, lines)
    geoms = [{
        "name": "skel",
        "geometry": line_set,
    }]

    vis.draw(geoms,
            bg_color=(0.8, 0.9, 0.9, 1.0),
            show_ui=True,
            width=1080,
            height=1080)
    return



if __name__ == '__main__':
    np_points = [
        [0,0,0],
        [10,10,10],
        [20,20,20],
        [20,30,20],
        [20,30,30],
        [30,30,30]
    ]

    np_lines = [
        [0,1],
        [1,2],
        [2,3],
        [3,4],
        [4,5],
    ]
    geoms = create_scene(np_points, np_lines)

    vis.draw(geoms,
             bg_color=(0.8, 0.9, 0.9, 1.0),
             show_ui=True,
             width=1080,
             height=1080)
