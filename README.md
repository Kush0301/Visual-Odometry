# MONOCULAR VISUAL ODOMETRY

### Introduction:

This project was aimed at understanding the front end method of estimating the motion of a calibrated camera mounted on a mobile platform. Estimating the location of a platform with respect to its surroundings and the path that it traversed is a key module in a Navigation Stack. Many sensors can be used to extract this particular information, camera being one of them. Cameras are one of the most widely used sensors for estimating the pose of a mobile platform. Visual Odometry is a process through which we can recover the relative translation and rotation of the calibrated camera using images that were recored by this particular camera.

### Method Proposed:

1. First image was processed and features were extracted using FAST.
2. Second image was processed and the features were tracked and determined using Kanade Lucas Tomasi feature tracking algorithm
3. Using the tracked correspondences the essential matrix was calculated.
4. The essential matrix was decomposed to recover the relative translation and rotation.
5. Using the tracked correspondences, relative translation and rotation, calibration matrix, triangulation is performed to obtain a 3D point cloud.
6. Process the third image, follow steps and track features using the second image as reference.
7. Perform steps 3-5
8. Relative scale is calculated by taking the mean of the distances using the two point clouds between matched keypoints between two sets of subsequent images.
9. Update translation and rotation.
10. Repeat steps 6-10. 

### Results

Calculation of absolute scale was avoided as ground truth will not available all the time. The result below is obtained on the basis of relative scale. 


![Result](https://github.com/Kush0301/Visual-Odometry/blob/master/visual_result.png?raw=true)
