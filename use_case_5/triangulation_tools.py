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
    # print(err)

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


# Add camera noise to the camera coordinates
def add_camera_noise(x1, x2, r, noise_values):
    # Pixel width - camera parameter
    pxWidth = 5.86e-6

    # Round the camera coordinates to the lowest integer
    left_cx_aux = np.round(x1[0] / pxWidth - np.round(x1[0] / pxWidth), 1)
    left_cy_aux = np.round(x1[1] / pxWidth - np.round(x1[1] / pxWidth), 1)

    right_cx_aux = np.round(x2[0] / pxWidth - np.round(x2[0] / pxWidth), 1)
    right_cy_aux = np.round(x2[1] / pxWidth - np.round(x2[1] / pxWidth), 1)

    # Select one of the ten values stored at random
    noise_index = np.random.randint(
        len(noise_values["left"][left_cx_aux][left_cy_aux][r])
    )

    # Retrieve the selected value for both cameras
    left_error_x, left_error_y = np.array(
        noise_values["left"][left_cx_aux][left_cy_aux][r][noise_index]
    )
    right_error_x, right_error_y = np.array(
        noise_values["right"][right_cx_aux][right_cy_aux][r][noise_index]
    )

    # Application of error
    left_cx_est = (np.round(x1[0] / pxWidth) + left_error_x) * pxWidth
    left_cy_est = (np.round(x1[1] / pxWidth) + left_error_y) * pxWidth

    right_cx_est = (np.round(x2[0] / pxWidth) + right_error_x) * pxWidth
    right_cy_est = (np.round(x2[1] / pxWidth) + right_error_y) * pxWidth

    return np.array([left_cx_est, left_cy_est]), np.array([right_cx_est, right_cy_est])


# triangulate a single point
def get_triangulated_point(
    point, P1, P2, P1_est, P2_est, noise_values=None, noise=False
):
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

    if noise:
        r = np.round(np.sqrt((point[0] ** 2) + (point[1] ** 2) + (point[2] ** 2)))
        x1, x2 = add_camera_noise(x1, x2, r, noise_values)

    # Call openCV function
    XestOpenCV = cv2.triangulatePoints(P1_est, P2_est, x1, x2).reshape((1, 4)).flatten()

    # Denormalization
    if XestOpenCV[3] != 0:
        XestOpenCV /= XestOpenCV[3]

    # Rearrange point to point cloud format
    XestOpenCV = [XestOpenCV[2], XestOpenCV[0], XestOpenCV[1]]

    return XestOpenCV
