import numpy as np
import cv2


# Obtain the realistic calibration matrices
def get_estimated_calibration_matrix(
    x=0.6, y=0, z=0, yaw=False, pitch=False, roll=False, num_pixels=1
):
    # focal length in [m]
    f1 = np.float64(16e-3)

    # intrisic camera matrices
    K1 = np.array([[f1, 0, 0], [0, f1, 0], [0, 0, 1]])
    K2 = K1.copy()

    # translation matrices
    # camera coordinates in [m]
    camera1T = np.array([[-0.6, 0, 0]]).T
    camera2T = np.array([[x, y, z]]).T

    # rotation matrices
    camera1R = np.identity(3)

    err = num_pixels * 0.024

    a2 = err * np.pi / 180 if yaw else 0  # alpha, yaw [rad]
    b2 = err * np.pi / 180 if pitch else 0  # beta, pitch [rad]
    g2 = err * np.pi / 180 if roll else 0  # gamma, roll [rad]

    camera2R = np.array(
        [
            [
                np.cos(a2) * np.cos(b2),
                np.cos(a2) * np.sin(b2) * np.sin(g2) - np.sin(a2) * np.cos(g2),
                np.cos(a2) * np.sin(b2) * np.cos(g2) + np.sin(a2) * np.sin(g2),
            ],
            [
                np.sin(a2) * np.cos(b2),
                np.sin(a2) * np.sin(b2) * np.sin(g2) + np.cos(a2) * np.cos(g2),
                np.sin(a2) * np.sin(b2) * np.cos(g2) - np.cos(a2) * np.sin(g2),
            ],
            [-np.sin(b2), np.cos(b2) * np.sin(g2), np.cos(b2) * np.cos(g2)],
        ]
    )

    # extrinsic camera matrices
    P1 = np.matmul(K1, np.concatenate((camera1R, -camera1T), axis=1))
    P2 = np.matmul(K2, np.concatenate((camera2R, -camera2T), axis=1))

    return [P1, P2]


# Obtain the ideal calibration matrices
def get_ideal_calibration_matrix():
    # focal length in [m]
    f1 = np.float64(16e-3)

    # intrisic camera matrices
    K1 = np.array([[f1, 0, 0], [0, f1, 0], [0, 0, 1]])

    K2 = K1.copy()

    # translation matrices
    # camera coordinates in [m]
    camera1T = np.array([[-0.6, 0, 0]]).T
    camera2T = np.array([[0.6, 0, 0]]).T

    # rotation matrices
    camera1R = np.identity(3)

    camera2R = np.identity(3)

    # extrinsic camera matrices
    P1 = np.matmul(K1, np.concatenate((camera1R, -camera1T), axis=1))
    P2 = np.matmul(K2, np.concatenate((camera2R, -camera2T), axis=1))

    return [P1, P2]


# triangulate a single point
def get_triangulated_point(point, P1, P2, P1_est, P2_est):
    # Rearrange point to fit to camera
    point = [point[1], point[2], point[0]]
    point = np.array(point).T
    point = np.hstack((point, np.ones((1,))))

    # get camera projetion coordinates
    x1 = np.matmul(P1, point)
    x2 = np.matmul(P2, point)

    # Normalization
    if x1[2] != 0:
        x1 /= x1[2]
    if x2[2] != 0:
        x2 /= x2[2]

    # Call openCV function
    XestOpenCV = (
        cv2.triangulatePoints(P1_est, P2_est, x1[0:2], x2[0:2])
        .reshape((1, 4))
        .flatten()
    )

    # Denormalization
    if XestOpenCV[3] != 0:
        XestOpenCV /= XestOpenCV[3]

    # Rearrange point to point cloud format
    XestOpenCV = [XestOpenCV[2], XestOpenCV[0], XestOpenCV[1]]

    return XestOpenCV
