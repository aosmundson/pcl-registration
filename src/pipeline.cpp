#include <iostream>
#include <numeric>
#include <boost/thread/thread.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/correspondence.h>
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/common/io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/filters/voxel_grid.h>


// --------------------
// -----Parameters-----
// --------------------
const float min_scale = 0.01f;
const int n_octaves = 3;
const int n_scales_per_octave = 4;
const float min_contrast = 0.001f;



// --------------
// -----Help-----
// --------------
void 
printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" [options] <file.pcd> <file.pcd>\n\n"
            << "Options:\n"
            << "-------------------------------------------\n"
            << "-h           this help\n"
            << "\n\n";
}

void 
setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}



// --------------
// -----Main-----
// --------------
int 
main (int argc, char** argv)
{
  // --------------------------------------
  // -----Parse Command Line Arguments-----
  // --------------------------------------
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    printUsage (argv[0]);
    return 0;
  }
  
  // -----------------------
  // -----Read pcd file-----
  // -----------------------
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>& source_cloud = *source_cloud_ptr;
  pcl::PointCloud<pcl::PointXYZ>& target_cloud = *target_cloud_ptr;
  Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
  std::vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument (argc, argv, "pcd");

  if (!pcd_filename_indices.empty ())
  {
    std::string src_filename = argv[pcd_filename_indices[0]];
    if (pcl::io::loadPCDFile (src_filename, source_cloud) == -1)
    {
      cerr << "Was not able to open file \""<<src_filename<<"\".\n";
      printUsage (argv[0]);
      return 0;
    }
    std::string tar_filename = argv[pcd_filename_indices[1]];
    if (pcl::io::loadPCDFile (tar_filename, target_cloud) == -1)
    {
      cerr << "Was not able to open file \""<<tar_filename<<"\".\n";
      printUsage (argv[0]);
      return 0;
    }
    scene_sensor_pose = Eigen::Affine3f (Eigen::Translation3f (source_cloud.sensor_origin_[0],
                                                               source_cloud.sensor_origin_[1],
                                                               source_cloud.sensor_origin_[2])) *
                        Eigen::Affine3f (source_cloud.sensor_orientation_);
  }
  else
  {
    cout << "\nNo *.pcd file given.\n\n";
    return 0;
  }
  
  // Downsample input clouds
  /*
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.setInputCloud(source_cloud_ptr);
  sor.filter(source_cloud);
  sor.setInputCloud(target_cloud_ptr);
  sor.filter(target_cloud);
  */

  // Estimate cloud normals
  cout << "Computing source cloud normals\n";
  pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
  pcl::PointCloud<pcl::PointNormal>::Ptr src_normals_ptr (new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>& src_normals = *src_normals_ptr;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n (new pcl::search::KdTree<pcl::PointXYZ>());
  ne.setInputCloud(source_cloud_ptr);
  ne.setSearchMethod(tree_n);
  ne.setRadiusSearch(0.05);
  ne.compute(*src_normals_ptr);
  cout << "Success\n";
  for(size_t i = 0;  i < src_normals.points.size(); ++i)
  {
      src_normals.points[i].x = source_cloud.points[i].x;
      src_normals.points[i].y = source_cloud.points[i].y;
      src_normals.points[i].z = source_cloud.points[i].z;
  }

  cout << "Computing target cloud normals\n";
  pcl::PointCloud<pcl::PointNormal>::Ptr tar_normals_ptr (new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>& tar_normals = *tar_normals_ptr;
  ne.setInputCloud(target_cloud_ptr);
  ne.compute(*tar_normals_ptr);
  cout << "Success\n";
  for(size_t i = 0;  i < tar_normals.points.size(); ++i)
  {
      tar_normals.points[i].x = target_cloud.points[i].x;
      tar_normals.points[i].y = target_cloud.points[i].y;
      tar_normals.points[i].z = target_cloud.points[i].z;
  }

  // Estimate the SIFT keypoints
  pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
  pcl::PointCloud<pcl::PointWithScale>::Ptr src_keypoints_ptr (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PointWithScale>& src_keypoints = *src_keypoints_ptr;
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal> ());
  sift.setSearchMethod(tree);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
  sift.setInputCloud(src_normals_ptr);
  sift.compute(src_keypoints);

  cout << "Found " << src_keypoints.points.size () << " SIFT keypoints in source cloud\n";
 
  pcl::PointCloud<pcl::PointWithScale>::Ptr tar_keypoints_ptr (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PointWithScale>& tar_keypoints = *tar_keypoints_ptr;
  sift.setInputCloud(tar_normals_ptr);
  sift.compute(tar_keypoints);

  cout << "Found " << tar_keypoints.points.size () << " SIFT keypoints in target cloud\n";
  
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer viewer ("3D Viewer");
  viewer.setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_cloud_color_handler (source_cloud_ptr, 255, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tar_cloud_color_handler (target_cloud_ptr, 0, 255, 255);
  viewer.addPointCloud (source_cloud_ptr, src_cloud_color_handler, "source cloud");
  viewer.addPointCloud (target_cloud_ptr, tar_cloud_color_handler, "target cloud");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");
  viewer.initCameraParameters ();
  setViewerPose (viewer, scene_sensor_pose);
  
  
  // -------------------------------------
  // -----Show keypoints in 3D viewer-----
  // -------------------------------------

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithScale> src_keypoints_color_handler (src_keypoints_ptr, 255, 0, 0);
  viewer.addPointCloud<pcl::PointWithScale> (src_keypoints_ptr, src_keypoints_color_handler, "source keypoints");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "source keypoints");

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithScale> tar_keypoints_color_handler (tar_keypoints_ptr, 0, 0, 255);
  viewer.addPointCloud<pcl::PointWithScale> (tar_keypoints_ptr, tar_keypoints_color_handler, "target keypoints");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "target keypoints");


  // Extract FPFH features from SIFT keypoints
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr src_keypoints_xyz (new pcl::PointCloud<pcl::PointXYZ>);                           
  pcl::copyPointCloud (src_keypoints, *src_keypoints_xyz);
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
  fpfh.setSearchSurface (source_cloud_ptr);
  fpfh.setInputCloud (src_keypoints_xyz);
  fpfh.setInputNormals (src_normals_ptr);
  fpfh.setSearchMethod (tree_n);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());
  pcl::PointCloud<pcl::FPFHSignature33>& src_features = *src_features_ptr;
  fpfh.setRadiusSearch(0.05);
  fpfh.compute(src_features);
  cout << "Computed " << src_features.size() << " FPFH features for source cloud\n";

  pcl::PointCloud<pcl::PointXYZ>::Ptr tar_keypoints_xyz (new pcl::PointCloud<pcl::PointXYZ>);                           
  pcl::copyPointCloud (tar_keypoints, *tar_keypoints_xyz);
  fpfh.setSearchSurface (target_cloud_ptr);
  fpfh.setInputCloud (tar_keypoints_xyz);
  fpfh.setInputNormals (tar_normals_ptr);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr tar_features_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());
  pcl::PointCloud<pcl::FPFHSignature33>& tar_features = *tar_features_ptr;
  fpfh.compute(tar_features);
  cout << "Computed " << tar_features.size() << " FPFH features for target cloud\n";
  
  

  // Estimate correspondences of FPFH features
  
  pcl::CorrespondencesPtr correspondences_ptr (new pcl::Correspondences);
  pcl::Correspondences correspondences = *correspondences_ptr;
  pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corr_est;
  corr_est.setInputSource(src_features_ptr);
  corr_est.setInputTarget(tar_features_ptr);
  corr_est.determineReciprocalCorrespondences(correspondences);



  // Reject bad correspondences


  //------Add features to visualizer------

  pcl::PointXYZ src_pt;
  pcl::PointXYZ corr_pt;
  for (size_t i; i < correspondences.size(); ++i)
  {
      src_pt.x = src_keypoints.points[correspondences[i].index_query].x;
      src_pt.y = src_keypoints.points[correspondences[i].index_query].y;
      src_pt.z = src_keypoints.points[correspondences[i].index_query].z;
      std::stringstream name1;
      name1 << "source point " << i;
      //viewer.addSphere<pcl::PointXYZ> (src_pt, 0.02, 255, 0, 0, name1.str());
  
      corr_pt.x = tar_keypoints.points[correspondences[i].index_match].x;
      corr_pt.y = tar_keypoints.points[correspondences[i].index_match].y;
      corr_pt.z = tar_keypoints.points[correspondences[i].index_match].z;
      std::stringstream name2;
      name2 << "corresponding point " << i;
      //viewer.addSphere<pcl::PointXYZ> (corr_pt, 0.02, 0, 0, 255, name2.str());
      cout << correspondences[i].distance << endl;
      if (correspondences[i].distance < 1000000000)
      {
          std::stringstream name3;
          name3 << "line" << i;
          viewer.addLine<pcl::PointXYZ, pcl::PointXYZ> (src_pt, corr_pt, 255, 0, 255, name3.str());
      }
  }

  //--------------------
  // -----Main loop-----
  //--------------------
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
    pcl_sleep(0.01);
  }
}
