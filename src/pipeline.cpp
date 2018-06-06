#include <iostream>
#include <numeric>
#include <boost/thread/thread.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/features/narf_descriptor.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/point_representation.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/correspondence.h>
#include <pcl/registration/correspondence_rejection_features.h>

typedef pcl::PointXYZ PointType;

// --------------------
// -----Parameters-----
// --------------------
float angular_resolution = 0.5f;
float support_size = 0.2f;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
bool setUnseenToMaxRange = false;
bool rotation_invariant = true;

// --------------
// -----Help-----
// --------------
void 
printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" [options] <scene.pcd>\n\n"
            << "Options:\n"
            << "-------------------------------------------\n"
            << "-r <float>   angular resolution in degrees (default "<<angular_resolution<<")\n"
            << "-c <int>     coordinate frame (default "<< (int)coordinate_frame<<")\n"
            << "-m           Treat all unseen points to max range\n"
            << "-s <float>   support size for the interest points (diameter of the used sphere - "
                                                                  "default "<<support_size<<")\n"
            << "-o <0/1>     switch rotational invariant version of the feature on/off"
            <<               " (default "<< (int)rotation_invariant<<")\n"
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

class NARFpr : public pcl::PointRepresentation<pcl::Narf36>
{
    public:
        NARFpr()
        {
            this->nr_dimensions_ = 36;
        }

        void copyToFloatArray (const pcl::Narf36 &narf, float *out) const
        {
            for(int i=0; i<36; ++i)
                out[i] = narf.descriptor[i];
        }
};


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
  if (pcl::console::find_argument (argc, argv, "-m") >= 0)
  {
    setUnseenToMaxRange = true;
    cout << "Setting unseen values in range image to maximum range readings.\n";
  }
  if (pcl::console::parse (argc, argv, "-o", rotation_invariant) >= 0)
    cout << "Switching rotation invariant feature version "<< (rotation_invariant ? "on" : "off")<<".\n";
  int tmp_coordinate_frame;
  if (pcl::console::parse (argc, argv, "-c", tmp_coordinate_frame) >= 0)
  {
    coordinate_frame = pcl::RangeImage::CoordinateFrame (tmp_coordinate_frame);
    cout << "Using coordinate frame "<< (int)coordinate_frame<<".\n";
  }
  if (pcl::console::parse (argc, argv, "-s", support_size) >= 0)
    cout << "Setting support size to "<<support_size<<".\n";
  if (pcl::console::parse (argc, argv, "-r", angular_resolution) >= 0)
    cout << "Setting angular resolution to "<<angular_resolution<<"deg.\n";
  angular_resolution = pcl::deg2rad (angular_resolution);
  
  // ------------------------------------------------------------------
  // -----Read pcd file or create example point cloud if not given-----
  // ------------------------------------------------------------------
  pcl::PointCloud<PointType>::Ptr source_cloud_ptr (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr target_cloud_ptr (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>& source_cloud = *source_cloud_ptr;
  pcl::PointCloud<PointType>& target_cloud = *target_cloud_ptr;
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
  
  // -----------------------------------------------
  // -----Create RangeImage from the PointCloud-----
  // -----------------------------------------------
  float noise_level = 0.0;
  float min_range = 0.0f;
  int border_size = 1;
  boost::shared_ptr<pcl::RangeImage> src_range_image_ptr (new pcl::RangeImage);
  boost::shared_ptr<pcl::RangeImage> tar_range_image_ptr (new pcl::RangeImage);
  pcl::RangeImage& src_range_image = *src_range_image_ptr;   
  pcl::RangeImage& tar_range_image = *tar_range_image_ptr;   
  src_range_image.createFromPointCloud (source_cloud, angular_resolution, pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
                                   scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
  tar_range_image.createFromPointCloud (target_cloud, angular_resolution, pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
                                   scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
  if (setUnseenToMaxRange)
    src_range_image.setUnseenToMaxRange ();
    tar_range_image.setUnseenToMaxRange ();
  
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer viewer ("3D Viewer");
  viewer.setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> src_range_image_color_handler (src_range_image_ptr, 255, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> tar_range_image_color_handler (tar_range_image_ptr, 0, 255, 255);
  viewer.addPointCloud (src_range_image_ptr, src_range_image_color_handler, "source range image");
  viewer.addPointCloud (tar_range_image_ptr, tar_range_image_color_handler, "target range image");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source range image");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target range image");
  //viewer.addCoordinateSystem (1.0f, "global");
  //PointCloudColorHandlerCustom<PointType> source_cloud_color_handler (source_cloud_ptr, 150, 150, 150);
  //viewer.addPointCloud (source_cloud_ptr, source_cloud_color_handler, "original point cloud");
  viewer.initCameraParameters ();
  setViewerPose (viewer, src_range_image.getTransformationToWorldSystem ());
  
  
  // --------------------------------
  // -----Extract NARF keypoints-----
  // --------------------------------
  pcl::RangeImageBorderExtractor range_image_border_extractor;
  pcl::NarfKeypoint narf_keypoint_detector;
  narf_keypoint_detector.setRangeImageBorderExtractor (&range_image_border_extractor);
  narf_keypoint_detector.setRangeImage (&src_range_image);
  narf_keypoint_detector.getParameters ().support_size = support_size;
  
  pcl::PointCloud<int> src_keypoint_indices;
  narf_keypoint_detector.compute (src_keypoint_indices);
  std::cout << "Found "<<src_keypoint_indices.points.size ()<<" key points in source cloud.\n";

  pcl::PointCloud<int> tar_keypoint_indices;
  narf_keypoint_detector.setRangeImage (&tar_range_image);
  narf_keypoint_detector.compute (tar_keypoint_indices);
  std::cout << "Found "<<tar_keypoint_indices.points.size ()<<" key points in target cloud.\n";
  
  
  // -------------------------------------
  // -----Show keypoints in 3D viewer-----
  // -------------------------------------
  pcl::PointCloud<pcl::PointXYZ>::Ptr src_keypoints_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>& src_keypoints = *src_keypoints_ptr;
  src_keypoints.points.resize (src_keypoint_indices.points.size ());
  src_keypoints.width = src_keypoint_indices.points.size();
  src_keypoints.height = 1;
  for (size_t i=0; i<src_keypoint_indices.points.size (); ++i)
    src_keypoints.points[i].getVector3fMap () = src_range_image.points[src_keypoint_indices.points[i]].getVector3fMap ();
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_keypoints_color_handler (src_keypoints_ptr, 0, 255, 0);
  viewer.addPointCloud<pcl::PointXYZ> (src_keypoints_ptr, src_keypoints_color_handler, "source keypoints");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "source keypoints");
 

  pcl::PointCloud<pcl::PointXYZ>::Ptr tar_keypoints_ptr (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>& tar_keypoints = *tar_keypoints_ptr;
  tar_keypoints.points.resize (tar_keypoint_indices.points.size ());
  tar_keypoints.width = tar_keypoint_indices.points.size();
  tar_keypoints.height = 1;
  for (size_t i=0; i<tar_keypoint_indices.points.size (); ++i)
    tar_keypoints.points[i].getVector3fMap () = tar_range_image.points[tar_keypoint_indices.points[i]].getVector3fMap ();
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tar_keypoints_color_handler (tar_keypoints_ptr, 0, 255, 0);
  viewer.addPointCloud<pcl::PointXYZ> (tar_keypoints_ptr, tar_keypoints_color_handler, "target keypoints");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "target keypoints");

  // ------------------------------------------------------
  // -----Extract NARF descriptors for interest points-----
  // ------------------------------------------------------
  std::vector<int> src_keypoint_indices2;
  src_keypoint_indices2.resize (src_keypoint_indices.points.size ());
  for (unsigned int i=0; i<src_keypoint_indices.size (); ++i) // This step is necessary to get the right vector type
    src_keypoint_indices2[i]=src_keypoint_indices.points[i];
  pcl::NarfDescriptor src_narf_descriptor (&src_range_image, &src_keypoint_indices2);
  src_narf_descriptor.getParameters ().support_size = support_size;
  src_narf_descriptor.getParameters ().rotation_invariant = rotation_invariant;
  pcl::PointCloud<pcl::Narf36> src_narf_descriptors;
  src_narf_descriptor.compute (src_narf_descriptors);
  cout << "Extracted "<<src_narf_descriptors.size ()<<" descriptors for "
                      <<src_keypoint_indices.points.size ()<< " keypoints in source cloud.\n";
  
  std::vector<int> tar_keypoint_indices2;
  tar_keypoint_indices2.resize (tar_keypoint_indices.points.size ());
  for (unsigned int i=0; i<tar_keypoint_indices.size (); ++i) // This step is necessary to get the right vector type
    tar_keypoint_indices2[i]=tar_keypoint_indices.points[i];
  pcl::NarfDescriptor tar_narf_descriptor (&tar_range_image, &tar_keypoint_indices2);
  tar_narf_descriptor.getParameters ().support_size = support_size;
  tar_narf_descriptor.getParameters ().rotation_invariant = rotation_invariant;
  pcl::PointCloud<pcl::Narf36>::Ptr tar_narf_descriptors_ptr (new pcl::PointCloud<pcl::Narf36>);
  pcl::PointCloud<pcl::Narf36>& tar_narf_descriptors = *tar_narf_descriptors_ptr;
  tar_narf_descriptor.compute (tar_narf_descriptors);
  cout << "Extracted "<<tar_narf_descriptors.size ()<<" descriptors for "
                      <<tar_keypoint_indices.points.size ()<< " keypoints in target cloud.\n";
  

  //------------------------------------------------------------------
  //------Use kd-tree nearest neighbor search on NARF descriptors------
  //-------------------------------------------------------------------

  pcl::KdTreeFLANN<pcl::Narf36> kdtree;
  kdtree.setPointRepresentation (boost::make_shared<NARFpr>());
  kdtree.setInputCloud(tar_narf_descriptors_ptr);
  int k = 5;
  std::vector<int> nkn_indices_i(k);
  std::vector<float> nkn_sq_dists_i(k);
  std::vector<int> nkn_indices(src_narf_descriptors.size());
  std::vector<float> nkn_sq_dists(src_narf_descriptors.size());
  for(size_t i = 0; i < src_narf_descriptors.size(); ++i)
  {
    cout << "Searching for " << k << " nearest neighbors of source descriptor " << i
        << ":\nx: " << src_narf_descriptors.points[i].x
        << "\ny: " << src_narf_descriptors.points[i].y
        << "\nz: " << src_narf_descriptors.points[i].z << "\n";
    kdtree.nearestKSearch (src_narf_descriptors.points[i], k, nkn_indices_i, nkn_sq_dists_i);
    nkn_indices[i] = nkn_indices_i[0];
    nkn_sq_dists[i] = nkn_sq_dists_i[0];
    for (size_t j = 0; j < k; ++j)
        cout << "index: " << nkn_indices_i[j]
            << "     squared distance: " << nkn_sq_dists_i[j] << "\n";
  }
  cout << "Results:\n";
  for(size_t i = 0; i < src_narf_descriptors.size(); ++i)
  {
      cout << "Source descriptor " << i << " corresponds to target descriptor "
          << nkn_indices[i]
          << " with a squared distance of " << nkn_sq_dists[i] << "\n";
  }


  //--------------------------------------
  //------Reject bad correspondences------
  //--------------------------------------

  // Create original correspondences object
  pcl::Correspondences original_correspondences;

  // fill original correspondences object with indices and distances
  for (size_t i; i < src_narf_descriptors.size(); ++i)
  {
      pcl::Correspondence correspondence(i, nkn_indices[i], nkn_sq_dists[i]);
      original_correspondences.push_back(correspondence);
  }

  // Create filtered correspondences object
  pcl::CorrespondencesPtr remaining_correspondences_ptr (new pcl::Correspondences);
  pcl::Correspondences remaining_correspondences = *remaining_correspondences_ptr;
  
  // Create correspondence rejector object
  pcl::registration::CorrespondenceRejectorFeatures rejector;
  // set feature representation
  // set input source
  // set input target
  // set input correspondences
  // get filtered correspondences / apply rejection (compare these two functions?)


  float avg_sq_dist = std::accumulate(nkn_sq_dists.begin(),nkn_sq_dists.end(),0.0)/nkn_sq_dists.size();
  cout << "Average nearest neighbor squared distance: " << avg_sq_dist << "\n";

  std::vector<int> src_corr_indices;
  std::vector<int> tar_corr_indices;

  float thresh = 0.5;

  for(size_t i = 0; i < src_narf_descriptors.size(); ++i)
  {
      if (nkn_sq_dists[i] < thresh)
      {
          pcl::Correspondence correspondence(i, nkn_indices[i], nkn_sq_dists[i]);
          remaining_correspondences.push_back(correspondence);
      }
  }
  cout << "Rejecting bad correspondences:\n";
  for(size_t i = 0; i < remaining_correspondences.size(); ++i)
  {
      cout << "Source descriptor " << remaining_correspondences[i].index_query << " corresponds to target descriptor "
          << remaining_correspondences[i].index_match
          << " with a squared distance of " << remaining_correspondences[i].distance << "\n";
  }

  //TODO: reject duplicate correspondences


  //--------------------------------------
  //------Add features to visualizer------
  //--------------------------------------

  pcl::PointXYZ src_pt;
  pcl::PointXYZ corr_pt;
  for (size_t i; i < remaining_correspondences.size(); ++i)
  {
      src_pt.x = src_narf_descriptors.points[remaining_correspondences[i].index_query].x;
      src_pt.y = src_narf_descriptors.points[remaining_correspondences[i].index_query].y;
      src_pt.z = src_narf_descriptors.points[remaining_correspondences[i].index_query].z;
      std::stringstream name1;
      name1 << "source point " << i;
      viewer.addSphere<pcl::PointXYZ> (src_pt, 0.02, 255, 0, 0, name1.str());
  
      corr_pt.x = tar_narf_descriptors.points[remaining_correspondences[i].index_match].x;
      corr_pt.y = tar_narf_descriptors.points[remaining_correspondences[i].index_match].y;
      corr_pt.z = tar_narf_descriptors.points[remaining_correspondences[i].index_match].z;
      std::stringstream name2;
      name2 << "corresponding point " << i;
      viewer.addSphere<pcl::PointXYZ> (corr_pt, 0.02, 0, 0, 255, name2.str());

      std::stringstream name3;
      name3 << "line" << i;
      viewer.addLine<pcl::PointXYZ, pcl::PointXYZ> (src_pt, corr_pt, 255, 0, 255, name3.str());
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
