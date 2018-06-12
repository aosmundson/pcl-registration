# pcl-registration

This package takes two point clouds as input arguments and attempts to align them using feature matching. The syntax for running the executable is 
```
./pipeline <source.pcd> <target.pcd>
```

First, the normals of each point cloud are estimated using a PCL **Normal Estimation** object. Then, using these normals as input, a PCL **SIFT Keypoint** object is used to compute the SIFT keypoints of each input cloud.

A PCL **FPFH Estimation** object is used to estimate the FPFH features at the keypoints of both clouds. Note that the search surface must be set to the original point cloud, but the input cloud is set to the keypoint cloud.

Once the keypoints and features have been estimated, they are used as input for the `computeInitialAlignment()` method, which is defined in the `include/pipeline.hpp` file. This method uses a PCL **Sample Consensus Initial Alignment** object to return a 4 by 4 transformation matrix. Finally, this matrix is applied to the source cloud to align it with the target cloud.

The program opens a PCL visualizer with two viewports. The left viewport displays the two input clouds and their SIFT keypoints, without any transformation. The right viewport displays the transformed source cloud and the original target cloud.

![Point Cloud Visualization](https://github.com/aosmundson/pcl-registration/blob/master/image/initial_alignment.png)

Parameters for keypoint detection and initial alignment can be tuned in the `include/pipeline.hpp` file.
