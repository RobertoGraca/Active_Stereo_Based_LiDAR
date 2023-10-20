import numpy as np
import os
import time

# Script to generate realistic point cloud based on Velodyne HDL-64E with only non-systematic error
if __name__ == "__main__":
    # Start timer
    start = int(time.time())

    # Iterate over all the original point clouds
    for frame in sorted(os.listdir("velodyne_1/")):
        # In case it stopped, it is easier to continue
        if frame in os.listdir("velodyne_2/"):
            continue

        # Identify the current point cloud
        print(f"Processing frame {frame} - {int(time.time())}")

        # Read point cloud content
        with open(f"velodyne_1/{frame}", "rb") as pc:
            content = np.fromfile(pc, dtype=np.float32).reshape(-1, 4)
            content = content[:, :3]
            pc.close()

        # Create new point cloud
        new_content = []

        # Iterate over the point cloud content
        for i in range(len(content)):
            # Divide the point into cartesian coordinates
            x = content[i][0]
            y = content[i][1]
            z = content[i][2]

            # Transform into spherical coordinates
            r = np.sqrt((x**2) + (y**2) + (z**2))
            teta = np.arctan(y / x)
            phi = np.arctan(z / (np.sqrt((x**2) + (y**2))))

            # Calculate  error value
            err = r + ((np.random.rand(1) - 0.5) / np.std(np.random.rand(2048)) * 10e-2)

            # Transform back to cartesian coordinate swith added error
            new_x = np.float32(err * np.cos(phi) * np.cos(teta))
            new_y = np.float32(err * np.cos(phi) * np.sin(teta))
            new_z = np.float32(err * np.sin(phi))

            # Add to the new point cloud
            new_content.append(new_x[0])
            new_content.append(new_y[0])
            new_content.append(new_z[0])
            new_content.append(0.0)

        # Write point cloud to file
        with open(f"velodyne_2/{frame}", "wb") as rpc:
            from array import array

            array("f", new_content).tofile(rpc)

    # Stop timer
    print(f"Took {int(time.time()) - start} seconds")
