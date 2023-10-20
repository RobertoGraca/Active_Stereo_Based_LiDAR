import numpy as np
import os
import cv2
import time
from triangulation_tools import *

# Script to generate point cloud obtained through triangulation with systematic error (calibration)
if __name__ == "__main__":
    # Start timer
    start = int(time.time())

    # Iterate over all the original point clouds
    for frame in sorted(os.listdir("velodyne_1/")):
        # In case it stopped, it is easier to continue
        if frame in os.listdir("velodyne_4/"):
            continue

        # Identify the current point cloud
        print(f"Processing frame {frame} - {int(time.time())}")

        # Read point cloud content
        with open(f"velodyne_1/{frame}", "rb") as pc:
            content = np.fromfile(pc, dtype=np.float32).reshape(-1, 4)
            content = content[:, :4]
            pc.close()

        # Setup ideal calibration matrices
        cal_matrix = get_ideal_calibration_matrix()
        P1 = cal_matrix[0]
        P2 = cal_matrix[1]

        # Setup estimated calibration matrices
        cal_matrix_est = get_estimated_calibration_matrix(
            0.6, 0, 0, True, True, True, 1
        )
        P1_est = cal_matrix_est[0]
        P2_est = cal_matrix_est[1]

        # Create new point cloud
        new_content = []

        # Iterate over the point cloud content
        for point in content:
            # Triangulate the current point
            triangulated_point = get_triangulated_point(point, P1, P2, P1_est, P2_est)

            # Add to the new point cloud
            new_content.append(triangulated_point[0])
            new_content.append(triangulated_point[1])
            new_content.append(triangulated_point[2])
            new_content.append(0.0)

        # Write point cloud to file
        with open(f"velodyne_4/{frame}", "wb") as rpc:
            from array import array

            array("f", new_content).tofile(rpc)

    # Stop timer
    print(f"Took {int(time.time()) - start} seconds")
