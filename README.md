# Active Stereo Based LiDAR

This project is about capturing a point cloud using a LiDAR based on an active stereo setup and compare it to an ideal LiDAR and a commercialy available LiDAR.

To accomplish that we used a [synthetic dataset](https://doi.org/10.5281/zenodo.7276691) captured by a LiDAR and transformed it according to requirements of six use cases:

1. Ideal LiDAR - The dataset is used as is.
2. CoTS ToF LiDAR - A non-systematic error is added to the dataset, based on a [Velodyne HDL64e](https://doi.org/10.1109/ACCESS.2020.3009680).
3. Ideal Active Stereo Based LiDAR - Ideal version of the proposed LiDAR. Points are obtained through triangulation.
4. Active Stereo Based LiDAR With Systematic Error - Introduction of Miscalibration
5. Active Stereo Based LiDAR With Non-Systematic Error - Introduction of Camera Noise
6. Active Stereo Based LiDAR With Systematic Error and Non-Systematic Error - Introduction of Miscalibration and Camera Noise

In this file, i am going over the steps taken to process the dataset into each of the use cases.

The original dataset is stored in binary files where the coordinates for each point are ordered as [x,y,z,i],[x,y,z,i]...
