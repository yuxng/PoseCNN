#include <sstream>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point32.h>
#include "synthesizer/PoseCNNMsg.h"
#include "synthesizer/synthesizer.hpp"

#define NUM_CLASSES 21
	
const std::string class_names[NUM_CLASSES] = {"002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can",
  "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box", "010_potted_meat_can",
  "011_banana", "019_pitcher_base", "021_bleach_cleanser", "024_bowl", "025_mug", "035_power_drill",
  "036_wood_block", "037_scissors", "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"};

Synthesizer* SYN;
std::vector<ros::Publisher> PUBs_poses(NUM_CLASSES);
std::vector<ros::Publisher> PUBs_points(NUM_CLASSES);

void callback(const synthesizer::PoseCNNMsg::ConstPtr& msg)
{
  int height = msg->height;
  int width = msg->width;
  int roi_num = msg->roi_num;
  int roi_channel = msg->roi_channel;
  float fx = msg->fx;
  float fy = msg->fy;
  float px = msg->px;
  float py = msg->py;
  float znear = msg->znear;
  float zfar = msg->zfar;
  float factor = msg->factor;
  const float* rois = msg->rois.data();
  const float* poses = msg->poses.data();

  // convert label
  cv_bridge::CvImagePtr cv_label_ptr;
  cv_label_ptr = cv_bridge::toCvCopy(msg->label, sensor_msgs::image_encodings::MONO8);
  cv::Mat label;
  cv_label_ptr->image.convertTo(label, CV_32SC1);

  // convert depth
  cv_bridge::CvImagePtr cv_depth_ptr;
  cv_depth_ptr = cv_bridge::toCvCopy(msg->depth, sensor_msgs::image_encodings::MONO16);

  // allocate outputs
  std::vector<float> outputs(roi_num * 7);
  std::vector<float> outputs_icp(roi_num * 7);
  
  // allocate point cloud
  std::vector<std::vector<geometry_msgs::Point32> > output_points(NUM_CLASSES);

  // ICP
  float maxError = 0.01;
  SYN->refineDistance((int*)label.data, cv_depth_ptr->image.data, height, width, fx, fy, px, py, 
    znear, zfar, factor, roi_num, roi_channel, rois, poses, outputs.data(), outputs_icp.data(), output_points, maxError);

  // publish the poses
  for (int i = 0; i < roi_num; i++)
  {
    int cls = int(rois[i * roi_channel + 1]);
    if (cls > 0)
    {
      geometry_msgs::PoseStamped pmsg;
      pmsg.pose.orientation.w = outputs[i * 7 + 0];
      pmsg.pose.orientation.x = outputs[i * 7 + 1];
      pmsg.pose.orientation.y = outputs[i * 7 + 2];
      pmsg.pose.orientation.z = outputs[i * 7 + 3];
      pmsg.pose.position.x = outputs[i * 7 + 4];
      pmsg.pose.position.y = outputs[i * 7 + 5];
      pmsg.pose.position.z = outputs[i * 7 + 6];  
      PUBs_poses[cls-1].publish(pmsg);

      sensor_msgs::PointCloud cmsg;
      cmsg.header.frame_id = "camera_link";
      cmsg.points = output_points[cls-1];
      PUBs_points[cls-1].publish(cmsg);
    }
  }
}

/**
 * This tutorial demonstrates simple sending of messages over the ROS system.
 */
int main(int argc, char **argv)
{
  ros::init(argc, argv, "synthesizer");
  ros::NodeHandle n;

  // initialize publishers
  for (int i = 0; i < NUM_CLASSES; i++)
  {
    std::string name_pose = "posecnn_pose_" + class_names[i];
    PUBs_poses[i] = n.advertise<geometry_msgs::PoseStamped>(name_pose, 1000);

    std::string name_point = "posecnn_points_" + class_names[i];
    PUBs_points[i] = n.advertise<sensor_msgs::PointCloud>(name_point, 1000);
  }

  // posecnn listener
  std::cout << argv[1] << std::endl;
  std::cout << argv[2] << std::endl;
  SYN = new Synthesizer (argv[1], argv[2]);
  SYN->setup(640, 480);
  ros::Subscriber sub = n.subscribe("posecnn_result", 1000, callback);

  ros::spin();

  return 0;
}
