#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


CURRENT_TEST_SCENE = 2
MODEL_PATH = '/home/robond/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/model.sav'
OUTPUT_FILENAME = "output_" + str(CURRENT_TEST_SCENE) + '.yaml'


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


'''Filters out noise'''
def filter_statistical_outliers(point_cloud, mean = 50, stdev = 0.5):
  outlier_filter = point_cloud.make_statistical_outlier_filter()
  outlier_filter.set_mean_k(mean)
  outlier_filter.set_std_dev_mul_thresh(stdev)
  return outlier_filter.filter()

'''Downsample point cloud''' 
def do_voxel_grid_filter(point_cloud, LEAF_SIZE = 0.01):
  voxel_filter = point_cloud.make_voxel_grid_filter()
  voxel_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE) 
  return voxel_filter.filter()

''' This pipeline performs three pass-through filters, one in each cartesian axis '''
def get_region_of_interest(cloud):

  filtered_cloud_z = do_passthrough_filter(point_cloud = cloud, 
                                         name_axis = 'z', min_axis = 0.6, max_axis = 1.3)

  filtered_cloud_zx = do_passthrough_filter(point_cloud = filtered_cloud_z, 
                                         name_axis = 'x', min_axis = 0.3, max_axis = 1.0)

  filtered_cloud_zxy = do_passthrough_filter(point_cloud = filtered_cloud_zx, 
                                         name_axis = 'y', min_axis = -0.5, max_axis = 0.5)

  return filtered_cloud_zxy

''' Uses RANSAC plane segmentation to separate plane and not plane points
    Returns inliers (plane) and outliers (not plane) '''
def do_ransac_plane_segmentation(point_cloud, max_distance = 0.01):

  segmenter = point_cloud.make_segmenter()

  segmenter.set_model_type(pcl.SACMODEL_PLANE)
  segmenter.set_method_type(pcl.SAC_RANSAC)
  segmenter.set_distance_threshold(max_distance)

  # Obtain inlier indices and coefficients of the plane model / equation
  inlier_indices, coefficients = segmenter.segment()

  inliers = point_cloud.extract(inlier_indices, negative = False)
  outliers = point_cloud.extract(inlier_indices, negative = True)

  return inliers, outliers

''' Returns only the point cloud information at a specific range of a specific axis '''
def do_passthrough_filter(point_cloud, name_axis = 'z', min_axis = 0.6, max_axis = 1.1):
  pass_filter = point_cloud.make_passthrough_filter()
  pass_filter.set_filter_field_name(name_axis)
  pass_filter.set_filter_limits(min_axis, max_axis)
  return pass_filter.filter()

''' This pipeline separates the objects on the table from the given scene '''
def split_cloud(cloud):
  
  # Reduce noise by taking out statistical outliers
  reduced_noise_cloud = filter_statistical_outliers(point_cloud = cloud, mean = 20, stdev = 0.5)

  # Downsample the cloud as high resolution which comes with a computation cost
  downsampled_cloud = do_voxel_grid_filter(point_cloud = reduced_noise_cloud, LEAF_SIZE = 0.005)

  # Get only information in our region of interest as we don't care about the other parts
  roi_cloud = get_region_of_interest(downsampled_cloud)

  # Separate the table from everything else
  table_cloud, objects_cloud = do_ransac_plane_segmentation(point_cloud = roi_cloud, max_distance = 0.01)

  return objects_cloud, table_cloud

def get_clusters(cloud, tolerance, min_size, max_size):

  tree = cloud.make_kdtree()
  extraction_object = cloud.make_EuclideanClusterExtraction()

  extraction_object.set_ClusterTolerance(tolerance)
  extraction_object.set_MinClusterSize(min_size)
  extraction_object.set_MaxClusterSize(max_size)
  extraction_object.set_SearchMethod(tree)

  # Get clusters of indices for each cluster of points, each clusterbelongs to the same object
  # 'clusters' is effectively a list of lists, with each list containing indices of the cloud
  clusters = extraction_object.Extract()
  return clusters

''' Clusters is a list of lists. Each list containing indices of the cloud
    cloud is an array with each cell having three numbers corresponding to x, y, z position
    Returns list of [x, y, z, color] '''
def get_colored_clusters(clusters, cloud):
  
  # Get a random unique colors for each object
  number_of_clusters = len(clusters)
  colors = get_color_list(number_of_clusters)

  colored_points = []

  # Assign a color for each point, points with the same color belong to the same cluster
  for cluster_id, cluster in enumerate(clusters):
    for c, i in enumerate(cluster):
      x, y, z = cloud[i][0], cloud[i][1], cloud[i][2]
      color = rgb_to_float(colors[cluster_id])
      colored_points.append([x, y, z, color])
  
  return colored_points


''' This class holds information about the object in the scene and what to do with it '''
class PickPlaceObject:

  def __init__(self, object):
    
    # These are ros msgs
    self.name = String()
    self.arm = String()
    self.pick_pose = Pose()
    self.place_pose =  Pose()
    
    self.group = None 
    self.yaml_dict = None

    self.name.data = str(object.label)
    
    # Assign pick_pose to centroid of cluster
    points = ros_to_pcl(object.cloud).to_array()
    x, y, z = np.mean(points, axis = 0)[:3]
    self.pick_pose.position.x = np.asscalar(x) 
    self.pick_pose.position.y = np.asscalar(y)
    self.pick_pose.position.z = np.asscalar(z)

    print "Creating object:", self.name.data
    print "in position:"
    print self.pick_pose.position
    print "Not yet determined: arm, place_pose, group"

  def set_place(self, pick_list, dropbox_list):
    
    '''
    IMPORTANT: 
      The YAML file where pick_list and dropbox_list is retrieved should 
      have the following format respectively:
      object_list:
      - name: biscuits
        group: green
      - name: soap
        group: green
      - name: soap2
        group: red
      dropbox:
      - name: left
        group: red
        position: [0,0.71,0.605]
      - name: right
        group: green
        position: [0,-0.71,0.605]
    '''

    print "Setting group and place position for object:", self.name.data

    for object in pick_list:
      if object['name'] == self.name.data:
        self.group = object['group']
        break

    print "group: ", self.group

    for box in dropbox_list:
      if box['group'] == self.group:
        x, y, z = box['position']
        print "should be placed at position:", x, y, z
        self.place_pose.position.x = np.float(x) 
        self.place_pose.position.y = np.float(y)
        self.place_pose.position.z = np.float(z)        
        self.arm.data = box['name']
        break

    print "ARM:", self.arm.data
    print "To be placed at:"
    print self.place_pose.position

  def make_yaml_dict(self, test_scene):
    self.yaml_dict = make_yaml_dict(test_scene, self.arm, self.name, self.pick_pose, self.place_pose)
    print "Yaml dictionary created"

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    pcl_data = ros_to_pcl(pcl_msg)

    objects_cloud, table_cloud = split_cloud(pcl_data)

    colorless_cloud = XYZRGB_to_XYZ(objects_cloud)
    
    clusters = get_clusters(colorless_cloud, tolerance = 0.01, min_size = 200, max_size = 15000)

    print "Number of clusters:", len(clusters)
    
    colored_points = get_colored_clusters(clusters, colorless_cloud)
    
    # Create a cloud with each cluster of points having the same color
    clusters_cloud = pcl.PointCloud_PointXYZRGB()
    clusters_cloud.from_list(colored_points)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects =  pcl_to_ros(clusters_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)

# Exercise-3 TODOs:
    
    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(clusters):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = objects_cloud.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
        ros_cluster_cloud = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        # TODO: complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster_cloud, using_hsv=True)
        normals = get_normals(ros_cluster_cloud)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(colorless_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster_cloud
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    test_scene = Int32()
    test_scene.data = CURRENT_TEST_SCENE
    output = []


    # TODO: Get/Read parameters
    pick_list = rospy.get_param('/object_list')
    dropbox_list = rospy.get_param('/dropbox')


    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for object in object_list:

        # TODO: Get the PointCloud for a given object and obtain it's centroid
        pickPlaceObj = PickPlaceObject(object)

        # TODO: Create 'place_pose' for the object
        pickPlaceObj.set_place(pick_list, dropbox_list)

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        # Make a yaml dictionary given current test scene and all available information
        pickPlaceObj.make_yaml_dict(test_scene)
        # Add this to our output
        output.append(pickPlaceObj.yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        '''
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            response = pick_place_routine(test_scene, 
                                    pickPlaceObj.name, 
                                    pickPlaceObj.arm, 
                                    pickPlaceObj.pick_pose, 
                                    pickPlaceObj.place_pose)
            print "Response: ", response.success

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        '''
    # TODO: Output your request parameters into output yaml file
    send_to_yaml(OUTPUT_FILENAME, output)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous = True)
    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size = 1)
    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("pcl_objects", PointCloud2, queue_size = 1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size = 1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size = 1)

    # Initialize color_list
    get_color_list.color_list = [] 

    # TODO: Load Model From disk
    model = pickle.load(open(MODEL_PATH, 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    
    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
