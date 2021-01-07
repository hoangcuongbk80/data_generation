#include <iostream> 
#include <cstdlib> 
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/tf.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <gazebo_msgs/SetLinkState.h>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/GetModelState.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>


using namespace cv;
using namespace std;
using namespace Eigen;

class readrgbdNode
{
  public:
    readrgbdNode();
    virtual ~readrgbdNode();
    void subcribeTopics();
    void advertiseTopics();
    void depthCallback(const sensor_msgs::Image::ConstPtr& msg);
    void rgbCallback(const sensor_msgs::Image::ConstPtr& msg);
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    void cloudPublish();
    void depthToClould(cv::Mat &depth_img);
    void move_objects();
    void move_camera();
    void get_object_states();
    void downsampling();
    void segmentation();
    void load_models();
    void colorMap(int i, pcl::PointXYZRGB &point);
    void extract_masks();
    void final_check();
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_cloud;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>> init_models;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>> curr_models;
    
    std::string object_instances[10];
    std::string object_categories[10];

    bool object_in_scene[10];
    int now;

    std::string depth_topsub, rgb_topsub, cloud_topsub, cloud_toppub;
    std::string saved_rgb_dir, saved_depth_dir, saved_points_dir;
    std::string optical_frame_id;
    std::string data_dir;
    
    std::ofstream pose_save_file; 
   
    double fx, fy, cx, cy, depth_factor;
    float ds_factor, neighbor_dist, ground_dist;
    bool debug;
   
    tf::TransformListener tf_listener;

    cv::Mat mask_img;

  private:
   ros::NodeHandle nh_, nh_rgb, nh_depth, nh_cloud;
   ros::Subscriber depth_sub, rgb_sub, cloud_sub;
   ros::Publisher cloud_pub;
};

readrgbdNode::readrgbdNode()
{
  nh_ = ros::NodeHandle("~");
  nh_rgb = ros::NodeHandle("~");
  nh_depth = ros::NodeHandle("~");
  nh_cloud = ros::NodeHandle("~");
  
  nh_depth.getParam("debug", debug);
  nh_depth.getParam("depth_topsub", depth_topsub);
  nh_rgb.getParam("rgb_topsub", rgb_topsub);
  nh_cloud.getParam("cloud_topsub", cloud_topsub);
  nh_.getParam("cloud_toppub", cloud_toppub);

  nh_.getParam("data_dir", data_dir);
  nh_.getParam("saved_rgb_dir", saved_rgb_dir);
  nh_.getParam("saved_depth_dir", saved_depth_dir);
  nh_.getParam("saved_points_dir", saved_points_dir);

  nh_.getParam("optical_frame_id", optical_frame_id);

  nh_.getParam("fx", fx);
  nh_.getParam("fy", fy);
  nh_.getParam("cx", cx);
  nh_.getParam("cy", cy);
  nh_.getParam("downsample", ds_factor);
  nh_.getParam("neighbor_dist", neighbor_dist);
  nh_.getParam("depth_factor", depth_factor);
  nh_.getParam("ground_dist", ground_dist); 


  std::cerr << "depth topics sub: " << "\n" << depth_topsub << "\n";
  std::cerr << "rgb topics sub: " << "\n" << rgb_topsub << "\n";
  std::cerr << "cloud topics sub: " << "\n" << cloud_topsub << "\n";
  std::cerr << "save depth to: " << "\n" << saved_depth_dir << "\n";
  std::cerr << "save rgb to: " << "\n" << saved_rgb_dir << "\n";
  std::cerr << "save points to: " << "\n" << saved_points_dir << "\n";

  object_instances[0] = "object1_can"; object_instances[1] = "object2_box";
  object_instances[2] = "object3_banana"; object_instances[3] = "object4_bowl";
  object_instances[4] = "object5_mug"; object_instances[5] = "object6_screwdriver";
  object_instances[6] = "object7_clamp"; object_instances[7] = "object8_baseball";
  object_instances[8] = "object9_brick"; object_instances[9] = "object10_cup";

  object_categories[0] = "007_tuna_fish_can"; object_categories[1] = "008_pudding_box";
  object_categories[2] = "011_banana"; object_categories[3] = "024_bowl";
  object_categories[4] = "025_mug"; object_categories[5] = "044_flat_screwdriver";
  object_categories[6] = "051_large_clamp"; object_categories[7] = "055_baseball";
  object_categories[8] = "061_foam_brick"; object_categories[9] = "065-h_cups";

  now = 0;

  cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  scene_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

  load_models();
}

readrgbdNode::~readrgbdNode()
{
};

void readrgbdNode::subcribeTopics()
{
  depth_sub = nh_depth.subscribe (depth_topsub, 1, &readrgbdNode::depthCallback, this);
  //rgb_sub = nh_rgb.subscribe (rgb_topsub, 1, &readrgbdNode::rgbCallback, this);  
  //cloud_sub = nh_cloud.subscribe (cloud_topsub, 1, &readrgbdNode::cloudCallback, this);
}

void readrgbdNode::advertiseTopics()
{
  cloud_pub = nh_.advertise<sensor_msgs::PointCloud2> (cloud_toppub, 1);
}

void readrgbdNode::load_models()
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud;

  for(int i=0; i < 10; i++)
  {
    model_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::string model_path = data_dir + "ycb-models/" + object_categories[i] + "/" + "nontextured.ply";
    if(debug) std::cerr << "\n" << "Load model " << model_path;
    pcl::io::loadPLYFile<pcl::PointXYZRGB> (model_path, *model_cloud);
    init_models.push_back(*model_cloud);
  }
}

void readrgbdNode::depthToClould(cv::Mat &depth_img)
{
   scene_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
   pcl::PointXYZRGB point;
   for(int row=0; row < depth_img.rows; row++)
    {
       for(int col=0; col < depth_img.cols; col++)       
        {
          if(isnan(depth_img.at<ushort>(row, col))) continue;
          double depth = depth_img.at<ushort>(row, col) / depth_factor;
          point.x = (col-cx) * depth / fx;
          point.y = (row-cy) * depth / fy;
          point.z = depth;
          int id = mask_img.at<uchar>(row, col); 
          point.b = id; point.g = id; point.r = id;
          scene_cloud->push_back(point);
        }
    }
}

void readrgbdNode::depthCallback (const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr bridge;

  try
  {
    bridge = cv_bridge::toCvCopy(msg, "32FC1");
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Failed to transform depth image.");
    return;
  }

  cv::Mat depth_img;
  depth_img = bridge->image.clone();
  depth_img.convertTo(depth_img, CV_16UC1, 1000.0); //Asus
  /* cv::imshow("depth", depth_img);
  cv::waitKey(3);
  cv::imwrite( saved_depth_dir, depth_img ); */

  // Get object poses and transfer models to world TF
  get_object_states();  

  move_objects();
  sleep(2);

  tf::StampedTransform transform;
  Eigen::Affine3d Tcam_offset;

  tf_listener.lookupTransform("world", optical_frame_id, ros::Time(0), transform);
  tf::poseTFToEigen(transform, Tcam_offset);
  Eigen::Matrix4d m = Tcam_offset.matrix().inverse();
  for(int i=0; i < 10; i++) pcl::transformPointCloud (curr_models[i], curr_models[i], m);  
  extract_masks();
  //for(int i=0; i < 10; i++) *scene_cloud += curr_models[i];

  depthToClould(depth_img);
  
  tf_listener.lookupTransform(optical_frame_id, "world", ros::Time(0), transform);
  tf::poseTFToEigen(transform, Tcam_offset);
  m = Tcam_offset.matrix().inverse();
  pcl::transformPointCloud (*scene_cloud, *scene_cloud, m);

  final_check();
  
  // Publish
  pcl::PCLPointCloud2 cloud_filtered;
  sensor_msgs::PointCloud2 output;
  scene_cloud->header.frame_id = "world";
  pcl::toPCLPointCloud2(*scene_cloud, cloud_filtered);
  pcl_conversions::fromPCL(cloud_filtered, output);
  cloud_pub.publish(output);

  // Save pointcloud
  pcl::io::savePLYFileBinary(saved_points_dir, *scene_cloud);
  //std::cerr << "save: " << saved_points_dir << "\n";

  curr_models.clear();
}

void readrgbdNode::rgbCallback (const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr bridge;
  try
  {
    bridge = cv_bridge::toCvCopy(msg, "bgr8");    
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Failed to transform rgb image.");
    return;
  }
  cv::Mat rgb_image;
  rgb_image = bridge->image;
  cv::imshow("RGB image", rgb_image);
  cv::waitKey(3);
  cv::imwrite( saved_rgb_dir, rgb_image );
}

void readrgbdNode::colorMap(int i, pcl::PointXYZRGB &point)
{
  if (i == 0) // silver  
  {
    point.r = 192; point.g = 192; point.b = 192; 
  }
  else if (i == 1) //red
  {
    point.r = 255; point.g = 0; point.b = 0;
  }
  else if (i == 2) //green
  {
    point.r = 0; point.g = 255; point.b = 0;
  }
  else if ( i == 3) //blue
  { 
    point.r = 0; point.g = 0; point.b = 255;
  } 
  else if ( i == 4) //maroon
  {
    point.r = 128; point.g = 0; point.b = 0;

  }
  else if ( i == 5) //line
  {
    point.r = 0; point.g = 128; point.b = 0;
  }  
  else if ( i == 6) //navy
  {
    point.r = 0; point.g = 0; point.b = 128;
  }
  else if ( i == 7) //yellow
  {
    point.r = 255; point.g = 255; point.b = 0;
  }
  else if ( i == 8) //magenta
  {
    point.r = 255; point.g = 0; point.b = 255;
  }
  else if ( i == 9) //cyan
  {
    point.r = 0; point.g = 255; point.b = 255;
  }    
  else if ( i == 10) //olive
  {
    point.r = 128; point.g = 128; point.b = 0;
  }
  else if ( i == 11) //purple
  {
    point.r = 128; point.g = 0; point.b = 128;
  } 
    
  else if ( i == 12) //teal
  {
    point.r = 0; point.g = 128; point.b = 128;
  }
    
  else if ( i == 13) 
  {
    point.r = 92; point.g = 112; point.b = 92;
  }
  else if ( i == 14) //brown
  {
    point.r = 165; point.g = 42; point.b = 42;
  }    
  else //silver
  {
    point.r = 192; point.g = 192; point.b = 192;
  }                   
}

void readrgbdNode::extract_masks()
{
  mask_img = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
  cv::Mat depth_min = cv::Mat::zeros(cv::Size(640, 480), CV_32F);

  pcl::PointXYZRGB point;

  for(int i=0; i < 10; i++)
  {
    for(int k=0; k<curr_models[i].size(); k++)
    {
        point = curr_models[i].points[k];
        int col = cx + (int) (point.x*fx / point.z);
        int row = cy + (int) (point.y*fy / point.z);
        if(col > 640 || col < 0) continue;
        if(row > 480 || row < 0) continue;
        if(point.z < depth_min.at<float>(row, col) || depth_min.at<float>(row, col)==0)
          mask_img.at<uchar>(row, col) = i+1;
    }
  }
  
  //cv::imshow("mask", mask_img);
  //cv::waitKey(3);
}

void readrgbdNode::final_check()
{
  for(int k=0; k<scene_cloud->size(); k++)
  {
    if(scene_cloud->points[k].z > ground_dist)
    {
      if(scene_cloud->points[k].r == 0)
      {
        scene_cloud->points[k].z = 0;
      }
      else 
      {
        if(debug) colorMap(scene_cloud->points[k].r, scene_cloud->points[k]);
        continue;
      }
    }
    else if(scene_cloud->points[k].r > 0)
    {
      scene_cloud->points[k].b = 0; 
      scene_cloud->points[k].g = 0; 
      scene_cloud->points[k].r = 0;
    }
    if(debug) colorMap(scene_cloud->points[k].r, scene_cloud->points[k]);
  }
}

void readrgbdNode::downsampling()
{
  // Create the filtering object
  pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  sor.setInputCloud (scene_cloud);
  sor.setLeafSize (ds_factor, ds_factor, ds_factor);
  sor.filter (*scene_cloud);
  if(debug) std::cerr << "Num Points after downsanpling: " << scene_cloud->size() << "\n";
}

void readrgbdNode::segmentation()
{
  
  
    for(int k=0; k<scene_cloud->size(); k++)
    {
      if(scene_cloud->points[k].z < ground_dist)
      {
          scene_cloud->points[k].r=192; scene_cloud->points[k].g=192; scene_cloud->points[k].b=192;
          continue;
      }

      float min_dist = 1000;
      for(int i=0; i < 10; i++)
      {
        if(!object_in_scene[i]) continue;

        pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(curr_models[i], *target_cloud);

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(target_cloud);
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        pcl::PointXYZ point;
        point.x = scene_cloud->points[k].x; point.y = scene_cloud->points[k].y; point.z = scene_cloud->points[k].z;

        if (kdtree.nearestKSearch(point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
          if(pointNKNSquaredDistance[0] < min_dist)
          {
            colorMap(i, scene_cloud->points[k]);
            min_dist = pointNKNSquaredDistance[0];
            //scene_cloud->points[k].r=20*i; scene_cloud->points[k].g=255-20*i; scene_cloud->points[k].b=0;
          }
        }
      } 
    }
}

void readrgbdNode::get_object_states()
{
  std::string pose_save_path = data_dir + "object_poses/" + std::to_string(now) + ".txt";
  pose_save_file.open(pose_save_path);
  pose_save_file << "object x y z qx qy qz qw";
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_cloud;

  for(int i=0; i < 10; i++)
  {
    model_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    /* if(!object_in_scene[i])
    {
      curr_models.push_back(*model_cloud);
      continue;
    } */ 
    gazebo_msgs::GetModelState getmodelstate;
    getmodelstate.request.model_name = object_instances[i];
    getmodelstate.request.relative_entity_name = "world";

    ros::ServiceClient client = nh_.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state"); 
    if (!client.call(getmodelstate))
    {
      ROS_ERROR("Failed to call service ");
    }
    geometry_msgs::Pose pose;
    pose = getmodelstate.response.pose;

    Eigen::Affine3d transform_i = Eigen::Affine3d::Identity();
    tf::poseMsgToEigen(pose, transform_i);
    pcl::transformPointCloud (init_models[i], *model_cloud, transform_i);
    curr_models.push_back(*model_cloud);

    pose_save_file << "\n" << object_categories[i];
    pose_save_file << " " << std::fixed << std::setprecision(4) << pose.position.x << " ";
    pose_save_file << " " << std::fixed << std::setprecision(4) << pose.position.y << " ";
    pose_save_file << " " << std::fixed << std::setprecision(4) << pose.position.z << " ";
    pose_save_file << " " << std::fixed << std::setprecision(4) << pose.orientation.x << " ";
    pose_save_file << " " << std::fixed << std::setprecision(4) << pose.orientation.y << " ";
    pose_save_file << " " << std::fixed << std::setprecision(4) << pose.orientation.z << " ";
  }
  pose_save_file.close();
  if(debug) std::cerr << "Save " << pose_save_path << "\n";
  now++;
}

void readrgbdNode::move_objects()
{
  float max_r = 0.3; 
  float rand_x, rand_y;

  for(int i=0; i < 10; i++)
  {
    object_in_scene[i] = true;
    int rand_i = rand();
    if(rand_i%9 < 5)
    {
      rand_x = i*2;
      rand_y = rand_x;
      object_in_scene[i] = false;
    }
    else if(rand_i%9==5)
    {
      rand_x = static_cast <float> (rand_i) / static_cast <float> (RAND_MAX / max_r);
      rand_y = static_cast <float> (rand_i) / static_cast <float> (RAND_MAX / max_r);
    }
    else if(rand_i%9==6)
    {
      rand_x = -1 * static_cast <float> (rand_i) / static_cast <float> (RAND_MAX / max_r);
      rand_y = -1 * static_cast <float> (rand_i) / static_cast <float> (RAND_MAX / max_r);
    }
    else if(rand_i%9==7)
    {
      rand_x = -1 * static_cast <float> (rand_i) / static_cast <float> (RAND_MAX / max_r);
      rand_y = static_cast <float> (rand_i) / static_cast <float> (RAND_MAX / max_r);
    }
    else if(rand_i%9==8)
    {
      rand_x = static_cast <float> (rand_i) / static_cast <float> (RAND_MAX / max_r);
      rand_y = -1 * static_cast <float> (rand_i) / static_cast <float> (RAND_MAX / max_r);
    }

    geometry_msgs::Pose start_pose;
    start_pose.position.x = rand_x;
    start_pose.position.y = rand_y;
    start_pose.position.z = 0;
    start_pose.orientation.x = 0.0;
    start_pose.orientation.y = 0.0;
    start_pose.orientation.z = 0.0;
    start_pose.orientation.w = 0.0;

    //---------------------------------------------------------------------
    gazebo_msgs::SetModelState setmodelstate;
    gazebo_msgs::ModelState modelstate;
    modelstate.model_name = object_instances[i]; 
    modelstate.reference_frame = "world";
    modelstate.pose = start_pose;
    setmodelstate.request.model_state = modelstate;

    ros::ServiceClient client = nh_.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state"); 
    if (!client.call(setmodelstate))
    {
      ROS_ERROR("Failed to call service ");
    }
  }
}

void readrgbdNode::move_camera()
{

  geometry_msgs::Pose start_pose;
  start_pose.position.x = 0.2;
  start_pose.position.y = 0;
  start_pose.position.z = 0;
  start_pose.orientation.x = 0.0;
  start_pose.orientation.y = 0.0;
  start_pose.orientation.z = 0.0;
  start_pose.orientation.w = 0.0;

  geometry_msgs::Twist start_twist;
  start_twist.linear.x = 1.1;
  start_twist.linear.y = 0;
  start_twist.linear.z = 0;
  start_twist.angular.x = 0.0;
  start_twist.angular.y = 0.0;
  start_twist.angular.z = 0.0;
  //---------------------------------------------------------------------
  gazebo_msgs::SetLinkState setlinkstate;
  gazebo_msgs::LinkState linkstate;
  linkstate.link_name = "camera_link"; 
  linkstate.reference_frame = "world";
  linkstate.twist = start_twist;
  linkstate.pose = start_pose;
  setlinkstate.request.link_state = linkstate;

  ros::ServiceClient client = nh_.serviceClient<gazebo_msgs::SetLinkState>("/gazebo/set_link_state"); 
  if (client.call(setlinkstate))
  {
    ROS_INFO("BRILLIANT!!!");
    ROS_INFO("%f, %f",linkstate.pose.position.x, linkstate.pose.position.y);
  }
  else
  {
    ROS_ERROR("Failed to call service ");
  }
}

void readrgbdNode::cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{ 
  if(debug) std::cerr << "\n------------------------" << std::to_string(now) << "-------------------------\n";
  pcl::PCLPointCloud2* cloud2 = new pcl::PCLPointCloud2; 
  pcl_conversions::toPCL(*cloud_msg, *cloud2);
  
  cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2( *cloud2, *cloud_);
  // do something
  scene_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud(*cloud_, *scene_cloud);
  //convert to world
  tf::StampedTransform transform;
  Eigen::Affine3d Tcam_offset;

  tf_listener.lookupTransform(optical_frame_id, "world", ros::Time(0), transform);
  tf::poseTFToEigen(transform, Tcam_offset);
  Eigen::Matrix4d m = Tcam_offset.matrix().inverse();
  pcl::transformPointCloud (*scene_cloud, *scene_cloud, m);

  downsampling();
  get_object_states();
  
  tf_listener.lookupTransform("world", optical_frame_id, ros::Time(0), transform);
  tf::poseTFToEigen(transform, Tcam_offset);
  m = Tcam_offset.matrix().inverse();
  for(int i=0; i < 10; i++) pcl::transformPointCloud (curr_models[i], curr_models[i], m);  
  extract_masks();
  // Save pointcloud
  saved_points_dir = data_dir + "pointcloud/" + std::to_string(now) + ".ply";
  pcl::io::savePLYFileBinary(saved_points_dir, *scene_cloud);
  if(debug) std::cerr << "Save " << saved_points_dir << "\n";
  //segmentation();
  // Publish
  //scene_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
  for(int i=0; i < 10; i++) *scene_cloud += curr_models[i];

  pcl::PCLPointCloud2 cloud_filtered;
  sensor_msgs::PointCloud2 output;
  scene_cloud->header.frame_id = optical_frame_id;
  pcl::toPCLPointCloud2(*scene_cloud, cloud_filtered);
  pcl_conversions::fromPCL(cloud_filtered, output);
  cloud_pub.publish(output);

  curr_models.clear();
  //move_objects();
  //sleep(3);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "find_transform");
  readrgbdNode mainNode;
  mainNode.subcribeTopics();
  mainNode.advertiseTopics();
  ros::spin();
  return 0;
}