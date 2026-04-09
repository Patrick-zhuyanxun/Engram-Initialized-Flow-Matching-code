import math

import numpy as np


def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    mat = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = mat[0, 0]
    m01 = mat[0, 1]
    m02 = mat[0, 2]
    m10 = mat[1, 0]
    m11 = mat[1, 1]
    m12 = mat[1, 2]
    m20 = mat[2, 0]
    m21 = mat[2, 1]
    m22 = mat[2, 2]
    # symmetric matrix k
    k = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    k /= 3.0
    # quaternion is Eigen vector of k that corresponds to largest eigenvalue
    w, v = np.linalg.eigh(k)
    inds = np.array([3, 0, 1, 2])
    q1 = v[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def rotate6d_to_axis_angle(r6d):
    """
    r6d: np.ndarray, shape (N, 6)
    return: np.ndarray, shape (N, 3), axis-angle vectors
    """
    flag = 0
    if len(r6d.shape) == 1:
        r6d = r6d[None, ...]
        flag = 1

    a1 = r6d[:, 0:3]
    a2 = r6d[:, 3:6]

    # b1
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-6)

    # b2
    dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2_orth = a2 - dot_prod * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + 1e-6)

    # b3
    b3 = np.cross(b1, b2, axis=-1)

    rotation_matrix = np.stack([b1, b2, b3], axis=-1)  # shape: (N, 3, 3)

    axis_angle_list = []
    for i in range(rotation_matrix.shape[0]):
        quat = mat2quat(rotation_matrix[i])
        axis_angle = quat2axisangle(quat)
        axis_angle_list.append(axis_angle)

    axis_angle_array = np.stack(axis_angle_list, axis=0)  # shape: (N, 3)

    if flag == 1:
        axis_angle_array = axis_angle_array[0]

    return axis_angle_array


def axis_angle_to_rot6d(axis_angles: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle vectors to 6D rotation representation.

    Uses the Rodrigues formula to build a full rotation matrix, then returns
    the first two columns (col1 + col2) as the 6D representation — exactly the
    inverse of the `rotate6d_to_axis_angle` path used at eval time.

    Args:
        axis_angles: (N, 3) or (3,) axis-angle vectors where magnitude encodes angle.

    Returns:
        np.ndarray: (N, 6) or (6,) 6D rotation representation.
    """
    flag = 0
    if axis_angles.ndim == 1:
        axis_angles = axis_angles[None, :]
        flag = 1

    N = axis_angles.shape[0]
    rot6d_list = []

    for i in range(N):
        aa = axis_angles[i].astype(np.float64)  # (3,)
        angle = np.linalg.norm(aa)

        if angle < 1e-8:
            # Identity rotation → first two columns of I
            rot6d_list.append(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32))
            continue

        axis = aa / angle  # unit axis
        kx, ky, kz = axis
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1.0 - c

        # Rodrigues rotation matrix
        R = np.array(
            [
                [t * kx * kx + c, t * kx * ky - s * kz, t * kx * kz + s * ky],
                [t * kx * ky + s * kz, t * ky * ky + c, t * ky * kz - s * kx],
                [t * kx * kz - s * ky, t * ky * kz + s * kx, t * kz * kz + c],
            ],
            dtype=np.float32,
        )

        # 6D representation: first two columns concatenated
        rot6d = np.concatenate([R[:, 0], R[:, 1]])  # (6,)
        rot6d_list.append(rot6d)

    result = np.stack(rot6d_list, axis=0)  # (N, 6)

    if flag == 1:
        result = result[0]

    return result


def mat_to_rotate6d(abs_action):
    if len(abs_action.shape) == 2:
        return np.concatenate([abs_action[:3, 0], abs_action[:3, 1]], axis=-1)
    elif len(abs_action.shape) == 3:
        return np.concatenate([abs_action[:, :3, 0], abs_action[:, :3, 1]], axis=-1)
    else:
        raise NotImplementedError


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor
