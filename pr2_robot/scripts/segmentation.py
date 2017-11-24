#!/usr/bin/env python

# Import modules
from pcl_helper import *

# TODO: Define functions as required
# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # TODO: Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)

    # TODO: Voxel Grid Downsampling



    # TODO: Publish ROS messages
    pcl_objects_pub.publish(pcl_msg)
    #pcl_table_pub.publish(ros_cloud_table)
    #pcl_clusterCloud_pub.publish(ros_cluster_cloud)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous = True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size = 1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("pcl_objects", PointCloud2, queue_size = 1)
    #pcl_table_pub = rospy.Publisher("pcl_table", PointCloud2, queue_size = 1)
    #pcl_clusterCloud_pub = rospy.Publisher("cluster_cloud", PointCloud2, queue_size = 1)

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
