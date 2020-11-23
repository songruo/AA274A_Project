#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Pose2D
#from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker

nav_data = None
def nav_callback(data):
    global nav_data
    nav_data = data

vendor_marker_id = -1
def vendor_callback(data):
    global vendor_marker_id
    vendor_marker_id += 1

    vis1_pub = rospy.Publisher('vendor_marker_topic', Marker, queue_size=10)
    marker = Marker()

    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time()

    # IMPORTANT: If you're creating multiple markers, 
    #            each need to have a separate marker ID.
    marker.id = vendor_marker_id

    marker.type = 2 # sphere

    marker.pose.position.x = data.x
    marker.pose.position.y = data.y
    marker.pose.position.z = 0.0

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1

    marker.color.a = 1.0 # Don't forget to set the alpha!
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    
    #print(marker.pose.position.x, marker.pose.position.y)
    vis1_pub.publish(marker)
    #print('Vendor marker published!')

def publisher():
    vis_pub = rospy.Publisher('marker_topic', Marker, queue_size=10)
    rospy.init_node('marker_node', anonymous=True)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
    	#rospy.Subscriber('/cmd_nav', Pose2D, nav_callback)
        rospy.Subscriber('/goal_loc', Pose2D, nav_callback)
        rospy.Subscriber('/vendor_loc', Pose2D, vendor_callback)
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        marker.id = 0

        marker.type = 2 # sphere
	if nav_data:

            marker.pose.position.x = nav_data.x
            marker.pose.position.y = nav_data.y
            marker.pose.position.z = 0.0

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.color.a = 1.0 # Don't forget to set the alpha!
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        
            vis_pub.publish(marker)
            #print('Published marker!')
        
        rate.sleep()
    # rospy.spin()


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
