#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from std_msgs.msg import String
#from std_msgs.msg import Float32MultiArray
from asl_turtlebot.msg import DetectedObject, DetectedObjectList
import tf
import numpy as np
from numpy import linalg
from utils import wrapToPi
from planners import AStar, compute_smoothed_traj
from planners.tsp_solver import *
from grids import StochOccupancyGrid2D
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 1
    ALIGN = 2
    TRACK = 3
    PARK = 4
    STOP = 5
    CROSS = 6
    #MANUAL = 7

class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """
    def __init__(self):
        rospy.init_node('turtlebot_navigator', anonymous=True)
        self.mode = Mode.IDLE

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.vendor_queue = []
        self.energy = 65

        self.th_init = 0.0

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0,0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False
        self.vendors = {'apple': None, 'banana': None, 'broccoli': None, 'pizza': None, 'donut': None, 'home': None}
        #self.vendors = {'apple': (0.27, 1), 'banana': (1, 0.27), 'broccoli': (1.85, 2.8), 'pizza': (1.15, 1.65), 'donut': (2.2, 1.85), 'home': (3.3, 1.6)}
        
        self.intermediate_left = (2.5, 0.3)
        self.intermediate_right = (2.5, 2.7)
        
        # plan parameters
        self.plan_resolution =  0.1
        self.plan_horizon = 15
        self.fail_count = 0
        # Stop sign related parameters
        # Time to stop at a stop sign
        self.stop_time = rospy.get_param("~stop_time", 3.)
        # Minimum distance from a stop sign to obey it
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.4)
        # Time taken to cross an intersection
        self.crossing_time = rospy.get_param("~crossing_time", 3.5)
        self.crossing_vel = 0.125

        self.vendor_detector_dist = 0.4
        self.vendor_time = rospy.get_param("~vendor_stop_time", 5.)
        self.vendor_stop_start_time = rospy.get_rostime()

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.,0.]
        
        # Robot limits
        self.v_max = 0.2    # maximum velocity
        self.om_max = 0.4   # maximum angular velocity
	#self.v_max = rospy.get_param("~v_max")
	#self.om_max = rospy.get_param("~om_max")

        self.v_des = 0.12   # desired cruising velocity
        self.theta_start_thresh = 0.05   # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = 0.2     # threshold to be far enough into the plan to recompute it

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.2
        self.at_thresh = 0.02
        self.at_thresh_theta = 0.05 # cfg

        # trajectory smoothing
        self.spline_alpha = 0.1 #need to change in cfg
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        # heading controller parameters
        self.kp_th = 2.

        self.traj_controller = TrajectoryTracker(self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max)
        self.pose_controller = PoseController(0., 0., 0., self.v_max, self.om_max)
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        self.nav_planned_path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.nav_smoothed_path_pub = rospy.Publisher('/cmd_smoothed_path', Path, queue_size=10)
        self.nav_smoothed_path_rej_pub = rospy.Publisher('/cmd_smoothed_path_rejected', Path, queue_size=10)
        self.nav_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.marker_pub = rospy.Publisher('/vendor_loc', Pose2D, queue_size=10)
        self.goal_marker_pub = rospy.Publisher('/goal_loc', Pose2D, queue_size=10)
        self.hungry_pub = rospy.Publisher('/hungry', String, queue_size=10)
        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/map_metadata', MapMetaData, self.map_md_callback)
        rospy.Subscriber('/cmd_nav', Pose2D, self.cmd_nav_callback)

        # Stop sign detector
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        # Vendor detector
        rospy.Subscriber('/detector/objects', DetectedObjectList, self.vendor_detected_callback)

        rospy.Subscriber('/delivery_request', String, self.order_callback)

        print "finished init"
        
    def dyn_cfg_callback(self, config, level):
        rospy.loginfo("Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}, spline_alpha:{spline_alpha}".format(**config))
        self.pose_controller.k1 = config["k1"] #default: 0.8
        self.pose_controller.k2 = config["k2"] #default: 0.4
        self.pose_controller.k3 = config["k3"] #default: 0.4
	self.spline_alpha = config["spline_alpha"] #default: 0.15
        self.at_thresh_theta = config["at_thresh_theta"] #default: 0.05
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if data.x != self.x_g or data.y != self.y_g or data.theta != self.theta_g:
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan()

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x,msg.origin.position.y)

    def map_callback(self,msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if self.map_width>0 and self.map_height>0 and len(self.map_probs)>0:
            self.occupancy = StochOccupancyGrid2D(self.map_resolution,
                                                  self.map_width,
                                                  self.map_height,
                                                  self.map_origin[0],
                                                  self.map_origin[1],
                                                  10,
                                                  self.map_probs)
            if self.x_g is not None:
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan() # new map, need to replan

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        thetaleft = msg.thetaleft - 2*np.pi if msg.thetaleft > np.pi else msg.thetaleft
        thetaright = msg.thetaright - 2*np.pi if msg.thetaright > np.pi else msg.thetaright
        camera_theta = (thetaleft + thetaright)/2
        dist = msg.distance*np.cos(camera_theta)

        # if close enough and in nav mode, stop
        if dist > 0 and dist < self.stop_min_dist and self.mode == Mode.TRACK and msg.confidence > 0.8:
            self.init_stop_sign()
	    print('stop sign detected')
   
    def vendor_detected_callback(self, msg):
        '''
        callback for when the detector has found a vendor logo it is looking for
        '''
        msg = msg.ob_msgs[-1]
        print(msg.name, msg.distance)
        if msg.name in self.vendors.keys() and msg.distance < self.vendor_detector_dist and self.vendors[msg.name] == None:
            print("Vendor detected")
            thetaleft = msg.thetaleft - 2*np.pi if msg.thetaleft > np.pi else msg.thetaleft
            thetaright = msg.thetaright - 2*np.pi if msg.thetaright > np.pi else msg.thetaright
            camera_theta = (thetaleft + thetaright)/2
            dist = msg.distance
            self.vendors[msg.name] = (self.x+dist*np.cos(self.theta+camera_theta), self.y+dist*np.sin(self.theta+camera_theta))
            pub_msg = Pose2D()
            pub_msg.x, pub_msg.y = self.vendors[msg.name]
            self.marker_pub.publish(pub_msg)           
            self.vendors[msg.name] = self.snap_to_grid((self.x, self.y))
            


    def order_callback(self, msg):
        '''
        callback for when there is an order request
        '''
        vendorList = str(msg)[7:-1].split(',')
        if vendorList == ['']:
            self.vendor_queue.append(self.vendors['home'])
        else:
            locations = [self.vendors['home']]
            for v in vendorList:
                if v != '':
                    locations.append(self.vendors[v])
            state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
            state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
            edges, cost_matrix = tsp_solver(state_min, state_max, self.occupancy, self.plan_resolution, self.v_des, self.spline_alpha, self.traj_dt, locations)
            print(edges, cost_matrix)
            queue_temp = []
            cost_list = [0]
            for edge in edges[1:]:
                queue_temp.append(locations[edge[1]])
                cost_list.append(cost_list[-1]+cost_matrix[edge[0]][edge[1]])
            print(cost_list)
            energy_fill_times = 0
            while cost_list[-1] > self.energy:
                index = len(cost_list)-1
                while index >= 0 and cost_list[index] > self.energy:
                    index -= 1
                while index >= 0 and cost_list[index] + compute_path_cost(state_min, state_max, self.occupancy, self.plan_resolution, self.v_des, self.spline_alpha, self.traj_dt, self.vendors['donut'], locations[edges[index][1]]) > self.energy:
                    index -= 1
                queue_temp.insert(index+energy_fill_times, self.vendors['donut'])
                energy_fill_times += 1
                # update cost_list
                cost_list[index+1:] = cost_list[index+1:] - cost_list[index+1] + compute_path_cost(state_min, state_max, self.occupancy, self.plan_resolution, self.v_des, self.spline_alpha, self.traj_dt, self.vendors['donut'], locations[edges[index+1][1]])
            print(queue_temp)
            self.vendor_queue = queue_temp

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        self.nav_vel_pub.publish(vel_g_msg)

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """

        self.stop_sign_start = rospy.get_rostime()
        self.mode = Mode.STOP

    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.stop_time)

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """

        self.cross_start = rospy.get_rostime()
        self.mode = Mode.CROSS
        print("init crossing")

    def has_crossed(self):
        """ checks if crossing maneuver is over """

        return self.mode == Mode.CROSS and \
               rospy.get_rostime() - self.cross_start > rospy.Duration.from_sec(self.crossing_time)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        return linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.near_thresh

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
	if self.x_g == None or self.y_g == None or self.theta_g == None:
            return True
        if (self.vendor_queue and linalg.norm(np.array([self.x-self.vendor_queue[0][0], self.y-self.vendor_queue[0][1]])) < self.near_thresh) or (linalg.norm(np.array([self.x-self.intermediate_left[0], self.y-self.intermediate_left[1]])) < self.near_thresh) or (linalg.norm(np.array([self.x-self.intermediate_right[0], self.y-self.intermediate_right[1]])) < self.near_thresh):
            return linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.at_thresh
        else:
            return (linalg.norm(np.array([self.x-self.x_g, self.y-self.y_g])) < self.at_thresh and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta)

    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh)
        
    def close_to_plan_start(self):
        return (abs(self.x - self.plan_start[0]) < self.start_pos_thresh and abs(self.y - self.plan_start[1]) < self.start_pos_thresh)

    def snap_to_grid(self, x):
        return (self.plan_resolution*round(x[0]/self.plan_resolution), self.plan_resolution*round(x[1]/self.plan_resolution))

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i,0]
            pose_st.pose.position.y = traj[i,1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = 'map'
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        t = self.get_current_plan_time()

        if self.mode == Mode.PARK:
            V, om = self.pose_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.TRACK:
            V, om = self.traj_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.ALIGN:
            V, om = self.heading_controller.compute_control(self.x, self.y, self.theta, t)
        elif self.mode == Mode.CROSS:
            V = self.crossing_vel
            om = 0.
        else:
            V = 0.
            om = 0.

        cmd_vel = Twist()
        cmd_vel.linear.x = V
        cmd_vel.angular.z = om
        self.nav_vel_pub.publish(cmd_vel)

    def get_current_plan_time(self):
        t = (rospy.get_rostime()-self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """
        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo("Navigator: replanning canceled, waiting for occupancy map.")
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(state_min,state_max,x_init,x_goal,self.occupancy,self.plan_resolution)

        rospy.loginfo("Navigator: computing navigation plan")
        success =  problem.solve()
        # deal with goal that is too far from where we are
        if not success:
            rospy.loginfo("Planning failed")
            self.fail_count += 1
	    if self.fail_count > 3:
                self.fail_count = 0
                left = abs(self.y - self.intermediate_left[1])
                right = abs(self.y - self.intermediate_right[1])
                self.x_g = 2.5
                if left < right:
                    self.y_g = self.intermediate_left[1]
                else:
                    self.y_g = self.intermediate_right[1]
                rospy.loginfo("Goal position reset due to failed plan")
            return
        rospy.loginfo("Planning Succeeded")

        planned_path = problem.path
        

        # Check whether path is too short
        if len(planned_path) < 4:
            rospy.loginfo("Path too short to track")
            self.switch_mode(Mode.PARK)
            self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
            return

        # Smooth and generate a trajectory
        traj_new, t_new = compute_smoothed_traj(planned_path, self.v_des, self.spline_alpha, self.traj_dt)

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = self.current_plan_duration - self.get_current_plan_time()

            # Estimate duration of new trajectory
            th_init_new = traj_new[0,2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err/self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo("New plan rejected (longer duration than current plan)")
                self.publish_smoothed_path(traj_new, self.nav_smoothed_path_rej_pub)
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0,2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            if not self.mode == Mode.STOP:
                if not (self.mode == Mode.CROSS and not self.has_crossed()):
            	    self.switch_mode(Mode.ALIGN)
            return

        if not self.mode == Mode.STOP:
            if not (self.mode == Mode.CROSS and not self.has_crossed()):
                rospy.loginfo("Ready to track")
                self.switch_mode(Mode.TRACK)

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation,rotation) = self.trans_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print e
                pass

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                if self.vendors['home'] == None and self.x != 0 and self.y != 0:
                    self.vendors['home'] = self.snap_to_grid((self.x + 0.2, self.y))
                    msg = Pose2D()
                    msg.x, msg.y = self.vendors['home']
                    self.marker_pub.publish(msg)

                if self.vendor_queue and (rospy.get_rostime() - self.vendor_stop_start_time) > rospy.Duration.from_sec(self.vendor_time):
                    self.x_g = self.vendor_queue[0][0]
                    self.y_g = self.vendor_queue[0][1]
                    #start from home
                    if (linalg.norm(np.array([self.x-self.vendors['home'][0], self.y-self.vendors['home'][1]])) < self.at_thresh):
                        left = abs(self.y_g - self.intermediate_left[1])
                        right = abs(self.y_g - self.intermediate_right[1])
                        self.x_g = 2.5
                        if left < right:
                            self.y_g = self.intermediate_left[1]
                        else:
                            self.y_g = self.intermediate_right[1]
                    #return to home
                    elif ((self.x_g==self.vendors['home'][0]) and (self.y_g==self.vendors['home'][1])) and not (linalg.norm(np.array([self.x-self.intermediate_left[0], self.y-self.intermediate_left[1]])) < self.at_thresh or linalg.norm(np.array([self.x-self.intermediate_right[0], self.y-self.intermediate_right[1]])) < self.at_thresh) and self.x < 3: 
                        left = abs(self.y - self.intermediate_left[1])
                        right = abs(self.y - self.intermediate_right[1])
                        self.x_g = 2.5
                        if left < right:
                            self.y_g = self.intermediate_left[1]
                        else:
                            self.y_g = self.intermediate_right[1]
                    self.theta_g = 0
                    self.replan()

            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (rospy.get_rostime() - self.current_plan_start_time).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan() # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                if self.at_goal():
                    if self.vendor_queue and linalg.norm(np.array([self.x-self.vendor_queue[0][0], self.y-self.vendor_queue[0][1]])) < self.at_thresh:
                        self.vendor_stop_start_time = rospy.get_rostime()
                        self.vendor_queue.pop(0)
                    # forget about goal:
                    self.x_g = None
                    self.y_g = None
                    self.theta_g = None
                    self.switch_mode(Mode.IDLE)

            elif self.mode == Mode.STOP:
                # At a stop sign
                if self.has_stopped():
            	    self.init_crossing()
                else:
                    self.stay_idle()

            elif self.mode == Mode.CROSS:
                # Crossing an intersection
	        if self.at_goal():
                    self.mode = Mode.IDLE
                elif self.has_crossed():
                    print("crossed")
                    self.replan()
	        else:
                    print("crossing") 
                    cmd_vel = Twist()
                    cmd_vel.linear.x = self.crossing_vel
                    cmd_vel.angular.z = 0
                    self.nav_vel_pub.publish(cmd_vel)


            self.publish_control()

            # Publish goal marker
            if self.x_g and self.y_g:
            	goal = Pose2D()
            	goal.x, goal.y = self.x_g, self.y_g
            	self.goal_marker_pub.publish(goal)
            
            # Publish hungry message
            if self.vendors['donut'] and self.x_g == self.vendors['donut'][0] and self.y_g == self.vendors['donut'][1]:
                msg = String()
                msg.data = "I am so hungry! I have to grab some donuts before I starve to die!!!"
                self.hungry_pub.publish(msg) 

            rate.sleep()

if __name__ == '__main__':    
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
