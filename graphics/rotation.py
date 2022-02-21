import torch

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / ((quaternions * quaternions).sum(-1) + 1e-9)

    B = quaternions.shape[:-1]
    device = quaternions.device
    zeros = torch.zeros(B).to(device)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            zeros,
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            zeros,
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
            zeros,
            zeros,
            zeros,
            zeros,
            torch.ones(B).cuda().to(device),
        ),
        -1,
    )

    return o.reshape(quaternions.shape[:-1] + (4, 4))

def get_identity_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    B = quaternions.shape[:-1]
    device = quaternions.device

    o = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32).unsqueeze(dim=0)

    o = o.repeat(B[0],1,1).to(device)
    return o

def get_batch_translation_matrix(B, position) -> torch.Tensor:
    device = position.device
    zeros = torch.zeros(B).to(device)
    ones = torch.ones(B).to(device)

    m = torch.stack(
        (
            ones, zeros, zeros, position[:,0],
            zeros, ones, zeros, position[:,1],
            zeros, zeros, ones, position[:,2],
            zeros, zeros, zeros, ones,
        ),
        -1,
    )

    return m.reshape((B,4,4))