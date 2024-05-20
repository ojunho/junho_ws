#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Path
from std_msgs.msg import Int64MultiArray, Float64, Int64, Bool
from sensor_msgs.msg import Imu
from geometry_msgs.msg import  Vector3
from visualization_msgs.msg import Marker, MarkerArray
from pyproj import Proj, transform
from morai_msgs.msg import GPSMessage, CtrlCmd, EventInfo
from morai_msgs.srv import MoraiEventCmdSrv
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from utils import pathReader,findLocalPath,purePursuit,rotateLiDAR2GPS, CCW
from lidar_object_detection.msg import ObjectInfo
# from lidar_cam_fusion.msg import Float64Array2D
# from ultralytics_ros.msg import YoloResult

import tf
from math import *
import numpy as np

# 아이오닉 5 -> 조향값(servo_msg) 0일 때 직진 양수이면 좌회전 음수이면 우회전

class EgoStatus:
    def __init__(self):
        self.position = Vector3()
        self.heading = 0.0
        self.velocity = Vector3()


class PurePursuit:
    def __init__(self):
        
        rospy.init_node('pure_pursuit', anonymous=True)

        self.path_name = 'tunnel_test'

        # 속도 50, 30 구간을 나누기 위한 변수 
        self.speed_limit = 30


        # Publisher
        self.global_path_pub                = rospy.Publisher('/global_path', Path, queue_size=1) ## global_path publisher 
        self.local_path_pub                 = rospy.Publisher('/local_path', Path, queue_size=1)
        self.heading_pub                    = rospy.Publisher('/heading', Float64, queue_size=1)
        self.current_waypoint_pub           = rospy.Publisher('/current_waypoint', Int64, queue_size=1)
        self.ctrl_cmd_pub                   = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)

        self.lattice_obstacle_pub           = rospy.Publisher('/lattice_obstacle_marker_array', MarkerArray, queue_size=1)
        self.acc_obstacle_pub               = rospy.Publisher('/acc_obstacle_marker_array', MarkerArray, queue_size=1)
        self.dynamic_obstacle_pub           = rospy.Publisher('/dynamic_obstacle_marker_array', MarkerArray, queue_size=1)

        self.pure_pursuit_target_point_pub  = rospy.Publisher('/pure_pusuit_target_point', Marker, queue_size=1)
        self.curvature_target_point_pub     = rospy.Publisher('/curvature_target_point', Marker, queue_size=1)
        self.ego_marker_pub                 = rospy.Publisher('/ego_marker', Marker, queue_size=1)


        # Subscriber
        rospy.Subscriber("/gps", GPSMessage, self.gpsCB) ## Vehicle Status Subscriber 
        rospy.Subscriber("/imu", Imu, self.imuCB) ## Vehicle Status Subscriber


        self.status_msg   = EgoStatus()
        self.ctrl_cmd_msg = CtrlCmd()


        self.lattice_lidar_obstacle_info = []
        self.acc_lidar_obstacle_info = []
        self.rotary_lidar_obstacle_info = []

        self.lattice_obstacle_info = []
        self.acc_obstacle_info = []
        self.dynamic_obstacle_info = []

        self.yolo_bbox_size_list = []

        self.rotary_stop_cnt = 0
        self.is_rotary_entered = False

        self.is_lab_time_check_started = False
        self.is_lab_time_check_finished = False

        self.is_status = False
        self.is_gps = False
        self.is_imu = False
        self.euler_data = [0,0,0,0]
        self.quaternion_data = [0,0,0,0]

        self.steering_angle_to_servo_offset = 0.0 ## servo moter offset
        self.target_x = 0.0
        self.target_y = 0.0
        self.curvature_target_x = 0.0
        self.curvature_target_y = 0.0
        self.corner_theta_degree = 0.0

        self.motor_msg = 0.0
        self.servo_msg = 0.0
        self.brake_msg = 0.0

        # -- 차선 커맨드 변수 -- #
        self.lane_ctrl_cmd_motor_msg = 0.0
        self.lane_ctrl_cmd_servo_msg = 0.0
        self.lane_ctrl_cmd_brake_msg = 0.0

        self.lane_cmd_2_motor_msg = 0.0
        self.lane_cmd_2_servo_msg = 0.0
        self.lane_cmd_2_brake_msg = 0.0
        # ---------------------- #

        self.steering_offset = 0.015
        # self.steering_offset = 0.05 
        

        self.curve_servo_msg = 0.0
        self.curve_motor_msg = 0.0

        self.target_velocity_array = []

        ########traffic_stop_#######
        self.green_light_count = 0
        self.red_light_count = 0

        self.stopline_flag = False
        self.current_waypoint = 0

        # path 별로 traiffic 위치 변경
 
        self.traffic_stop_index_1 = -1
        self.traffic_stop_index_2 = 1840

       
        self.traffic_light_status = None
        
        self.selected_lane = 1

        self.lattice_distance_threshold = 0.0
        self.acc_distance_threshold = 0.0
        self.dynamic_obstacle_distance_threshold = 0.0

        self.proj_UTM = Proj(proj='utm', zone = 52, elips='WGS84', preserve_units=False)
        self.tf_broadcaster = tf.TransformBroadcaster()

        # ######################################## For Service ########################################
        # rospy.wait_for_service('/Service_MoraiEventCmd')
        # self.req_service = rospy.ServiceProxy('/Service_MoraiEventCmd', MoraiEventCmdSrv)
        # self.req = EventInfo()

        # # self.forward_mode()
        # #############################################################################################

        # Class
        path_reader = pathReader('path_maker') ## 경로 파일의 위치
        self.pure_pursuit = purePursuit() ## purePursuit import

        # Read path
        self.global_path, self.target_velocity_array = path_reader.read_txt(self.path_name+".txt") ## 출력할 경로의 이름

        rate = rospy.Rate(40) 
                                           
        while not rospy.is_shutdown():
            self.getEgoStatus()
            if self.is_status == True:

                self.ctrl_cmd_msg.longlCmdType = 2

                local_path, self.current_waypoint = findLocalPath(self.global_path, self.status_msg)

                self.pure_pursuit.getPath(local_path) 
                self.pure_pursuit.getEgoStatus(self.status_msg) 
                self.steering, self.target_x, self.target_y = self.pure_pursuit.steeringAngle()
                self.corner_theta_degree, self.curvature_target_x, self.curvature_target_y = self.pure_pursuit.estimateCurvature()


                self.motor_msg = self.cornerController()
                self.servo_msg = (self.steering + 2.7) * self.steering_offset

            
                self.visualizeTargetPoint()
                self.visualizeCurvatureTargetPoint()
                self.visualizeEgoMarker()

                self.visualizeLatticeObstacle()
                self.visualizeAccObstacle()
                self.visualizeDynamicObstacle()


                self.local_path_pub.publish(local_path)
                self.global_path_pub.publish(self.global_path)
                self.heading_pub.publish(self.status_msg.heading)
                self.current_waypoint_pub.publish(self.current_waypoint)
                

                ########################################################################################################################################################
                self.publishCtrlCmd(self.motor_msg, self.servo_msg, self.brake_msg)
                
                ########################################################################################################################################################

            rate.sleep()
###################################################################### Service Request  ######################################################################
    # option - 1 : ctrl_mode / 2 : gear / 4 : lamps / 6 : gear + lamps
    # gear - 1: P / 2 : R / 3 : N / 4 : D
##############################################################################################################################################################

    def forward_mode(self):
        self.req.option = 6
        self.req.gear = 4
        self.req.lamps.turnSignal = 0
        self.req.lamps.emergencySignal = 0
        response = self.req_service(self.req)
        self.yaw_rear = False

    def rear_mode(self):
        self.req.option = 2
        self.req.gear = 2
        response = self.req_service(self.req)
        self.yaw_rear = True

    def drive_left_signal(self):
        self.req.option = 6
        self.req.gear = 4
        self.req.lamps.turnSignal = 1
        response = self.req_service(self.req)

    
    def drive_right_signal(self) :
        self.req.option = 6
        self.req.gear = 4
        self.req.lamps.turnSignal = 2
        response = self.req_service(self.req)

    def emergency_mode(self) :
        self.req.option = 6
        self.req.gear = 4
        self.req.lamps.emergencySignal = 1
        response = self.req_service(self.req)

    def parking(self) :
        self.req.option = 6
        self.req.gear = 1
        self.req.lamps.turnSignal = 0
        response = self.req_service(self.req)

    def brake(self) :
        self.ctrl_cmd_msg.longlCmdType = 2
        self.motor_msg = 0.0
        self.servo_msg = 0.0
        self.brake_msg = 1.0
        self.publishCtrlCmd(self.motor_msg, self.servo_msg, self.brake_msg)

    
###################################################################### Call Back ######################################################################
    def getEgoStatus(self): ## Vehicle Status Subscriber 
        if self.is_gps == True and self.is_imu == True:
            self.status_msg.position.x = self.xy_zone[0] - 302459.942 # 313008.55819800857
            self.status_msg.position.y = self.xy_zone[1] - 4122635.537 # 4161698.628368007
            self.status_msg.position.z = 0.0
            self.status_msg.heading = self.euler_data[2] * 180/pi
            self.status_msg.velocity.x = self.motor_msg #self.velocity

            self.tf_broadcaster.sendTransform((self.status_msg.position.x, self.status_msg.position.y, self.status_msg.position.z),
                            tf.transformations.quaternion_from_euler(0, 0, (self.status_msg.heading)/180*pi),
                            rospy.Time.now(),
                            "base_link",
                            "map")
   
            self.is_status=True

        elif self.is_gps is False and self.is_imu is True:
            self.status_msg.heading = self.euler_data[2] * 180/pi
            self.is_status=False

        else:
            # print("Waiting for GPS & IMU")
            self.is_status=False


    def gpsCB(self, msg):
        if msg.status == 0: 
            self.current_waypoint = -1
            self.is_gps = False

        else:
            self.xy_zone = self.proj_UTM(msg.longitude, msg.latitude)
            
            self.tf_broadcaster.sendTransform((0, 0, 1.18),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            "gps",
                            "base_link")
            
            self.tf_broadcaster.sendTransform((1.44, 0, 1.24),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            "velodyne",
                            "base_link")
            
            self.is_gps = True


    def imuCB(self, msg):
        self.quaternion_data = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        self.euler_data = tf.transformations.euler_from_quaternion((msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w))

        self.tf_broadcaster.sendTransform((-0.08, 0.0, 1.18),
                        tf.transformations.quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "imu",
                        "base_link")

        self.is_imu = True

    def trafficlightCB(self, msg):
        self.red_light_count   = msg.data[0]
        self.green_light_count = msg.data[1]
        # print("Green:", self.green_light_count, 'Red:', self.red_light_count)

    def latticeLidarObjectCB(self, msg):
        self.lattice_lidar_obstacle_info = [[] for i in range(msg.objectCounts)]
        for i in range(msg.objectCounts):
            self.lattice_lidar_obstacle_info[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i], msg.lengthX[i], msg.lengthY[i], msg.lengthZ[i]]

    def accLidarObjectCB(self, msg):
        self.acc_lidar_obstacle_info = [[] for i in range(msg.objectCounts)]
        for i in range(msg.objectCounts):
            self.acc_lidar_obstacle_info[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i], msg.lengthX[i], msg.lengthY[i], msg.lengthZ[i]]

    def rotaryLidarObjectCB(self, msg):
        self.rotary_lidar_obstacle_info = [[] for i in range(msg.objectCounts)]
        for i in range(msg.objectCounts):
            self.rotary_lidar_obstacle_info[i] = [msg.centerX[i], msg.centerY[i], msg.centerZ[i], msg.lengthX[i], msg.lengthY[i], msg.lengthZ[i]]
    
    def fusionResultCB(self, msg):
        self.fusion_result_drum = [list(bbox.bbox)[4:8] for bbox in msg.bboxes if bbox.bbox[7] == 0]
        self.fusion_result_person = [list(bbox.bbox)[4:8] for bbox in msg.bboxes if bbox.bbox[7] == 1]

    def laneCtrlCmdCB(self, msg):
        self.lane_ctrl_cmd_motor_msg = msg.velocity
        self.lane_ctrl_cmd_servo_msg = msg.steering
        self.lane_ctrl_cmd_brake_msg = msg.brake

    def stopLineCB(self, msg):
        self.stopline_flag = msg.data
        # print(self.stopline_flag)

    def yoloResultCB(self, msg):
        detections_list = msg.detections.detections
        self.yolo_bbox_size_list = [0.0 for i in range(len(detections_list))]

        for i in range(len(detections_list)):
            self.yolo_bbox_size_list[i] = detections_list[i].bbox.size_x * detections_list[i].bbox.size_y
            
    




###################################################################### Function ######################################################################

    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        self.ctrl_cmd_msg.velocity = motor_msg
        self.ctrl_cmd_msg.steering = servo_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

    def setBrakeMsgWithNum(self, brake):
        self.brake_msg = brake
    
    def cornerController(self):
        if self.corner_theta_degree > 50:
            self.corner_theta_degree = 50

        motor_msg = -0.7 * self.corner_theta_degree + 49

        return motor_msg 

    def visualizeLatticeObstacle(self):
        lattice_obstacle_array = MarkerArray()

        for i in range(len(self.lattice_obstacle_info)):
            lattice_obstacle = Marker()
            lattice_obstacle.header.frame_id = "map"
            lattice_obstacle.id = i
            lattice_obstacle.type = lattice_obstacle.CYLINDER
            lattice_obstacle.action = lattice_obstacle.ADD
            lattice_obstacle.scale.x = self.lattice_distance_threshold * 2
            lattice_obstacle.scale.y = self.lattice_distance_threshold * 2
            lattice_obstacle.scale.z = 2.0
            lattice_obstacle.pose.orientation.w = 1.0
            lattice_obstacle.color.r = 1.0
            lattice_obstacle.color.g = 1.0
            lattice_obstacle.color.b = 1.0
            lattice_obstacle.color.a = 0.5 
            lattice_obstacle.pose.position.x = self.lattice_obstacle_info[i][0]
            lattice_obstacle.pose.position.y = self.lattice_obstacle_info[i][1]
            lattice_obstacle.pose.position.z = 0.0
            lattice_obstacle.lifetime = rospy.Duration(0.1)

            lattice_obstacle_array.markers.append(lattice_obstacle)

        self.lattice_obstacle_pub.publish(lattice_obstacle_array)

    def visualizeDynamicObstacle(self):
        dynamic_obstacle_array = MarkerArray()

        for i in range(len(self.dynamic_obstacle_info)):
            dynamic_obstacle = Marker()
            dynamic_obstacle.header.frame_id = "map"
            dynamic_obstacle.id = i
            dynamic_obstacle.type = dynamic_obstacle.CYLINDER
            dynamic_obstacle.action = dynamic_obstacle.ADD
            dynamic_obstacle.scale.x = self.dynamic_obstacle_distance_threshold * 2
            dynamic_obstacle.scale.y = self.dynamic_obstacle_distance_threshold * 2
            dynamic_obstacle.scale.z = 2.0
            dynamic_obstacle.pose.orientation.w = 1.0
            dynamic_obstacle.color.r = 0.0
            dynamic_obstacle.color.g = 1.0
            dynamic_obstacle.color.b = 0.0
            dynamic_obstacle.color.a = 0.5 
            dynamic_obstacle.pose.position.x = self.dynamic_obstacle_info[i][0]
            dynamic_obstacle.pose.position.y = self.dynamic_obstacle_info[i][1]
            dynamic_obstacle.pose.position.z = 0.0
            dynamic_obstacle.lifetime = rospy.Duration(0.1)

            dynamic_obstacle_array.markers.append(dynamic_obstacle)

        self.dynamic_obstacle_pub.publish(dynamic_obstacle_array)

    def visualizeAccObstacle(self):
        acc_obstacle_array = MarkerArray()

        for i in range(len(self.acc_obstacle_info)):
            acc_obstacle = Marker()
            acc_obstacle.header.frame_id = "map"
            acc_obstacle.type = acc_obstacle.CYLINDER
            acc_obstacle.action = acc_obstacle.ADD
            acc_obstacle.scale.x = self.acc_distance_threshold * 2
            acc_obstacle.scale.y = self.acc_distance_threshold * 2
            acc_obstacle.scale.z = 2.0
            acc_obstacle.pose.orientation.w = 1.0
            acc_obstacle.color.r = 1.0
            acc_obstacle.color.g = 0.0
            acc_obstacle.color.b = 0.0
            acc_obstacle.color.a = 0.5 
            acc_obstacle.pose.position.x = self.acc_obstacle_info[i][0]
            acc_obstacle.pose.position.y = self.acc_obstacle_info[i][1]
            acc_obstacle.pose.position.z = 0.0
            acc_obstacle.lifetime = rospy.Duration(0.1)

            acc_obstacle_array.markers.append(acc_obstacle)

        self.acc_obstacle_pub.publish(acc_obstacle_array)


    def visualizeTargetPoint(self):
        pure_pursuit_target_point = Marker()
        pure_pursuit_target_point.header.frame_id = "map"
        pure_pursuit_target_point.type = pure_pursuit_target_point.SPHERE
        pure_pursuit_target_point.action = pure_pursuit_target_point.ADD
        pure_pursuit_target_point.scale.x = 1.0
        pure_pursuit_target_point.scale.y = 1.0
        pure_pursuit_target_point.scale.z = 1.0
        pure_pursuit_target_point.pose.orientation.w = 1.0
        pure_pursuit_target_point.color.r = 1.0
        pure_pursuit_target_point.color.g = 0.0
        pure_pursuit_target_point.color.b = 0.0
        pure_pursuit_target_point.color.a = 1.0 
        pure_pursuit_target_point.pose.position.x = self.target_x
        pure_pursuit_target_point.pose.position.y = self.target_y
        pure_pursuit_target_point.pose.position.z = 0.0
        
        self.pure_pursuit_target_point_pub.publish(pure_pursuit_target_point)


    def visualizeCurvatureTargetPoint(self):
        curvature_target_point = Marker()
        curvature_target_point.header.frame_id = "map"
        curvature_target_point.type = curvature_target_point.SPHERE
        curvature_target_point.action = curvature_target_point.ADD
        curvature_target_point.scale.x = 1.0
        curvature_target_point.scale.y = 1.0
        curvature_target_point.scale.z = 1.0
        curvature_target_point.pose.orientation.w = 1.0
        curvature_target_point.color.r = 0.0
        curvature_target_point.color.g = 0.0
        curvature_target_point.color.b = 1.0
        curvature_target_point.color.a = 1.0 
        curvature_target_point.pose.position.x = self.curvature_target_x
        curvature_target_point.pose.position.y = self.curvature_target_y
        curvature_target_point.pose.position.z = 0.0
        
        self.curvature_target_point_pub.publish(curvature_target_point)


    def visualizeEgoMarker(self):
        ego_marker = Marker()
        ego_marker.header.frame_id = "map"
        ego_marker.type = ego_marker.MESH_RESOURCE
        ego_marker.mesh_resource = "package://pure_pursuit/stl/egolf.stl"
        ego_marker.mesh_use_embedded_materials = True
        ego_marker.action = ego_marker.ADD
        ego_marker.scale.x = 1.2
        ego_marker.scale.y = 1.2
        ego_marker.scale.z = 1.2
        ego_marker.pose.orientation.x = self.quaternion_data[0]
        ego_marker.pose.orientation.y = self.quaternion_data[1]
        ego_marker.pose.orientation.z = self.quaternion_data[2]
        ego_marker.pose.orientation.w = self.quaternion_data[3]
        ego_marker.color.r = 1.0
        ego_marker.color.g = 1.0
        ego_marker.color.b = 1.0
        ego_marker.color.a = 1.0
        ego_marker.pose.position.x = self.status_msg.position.x
        ego_marker.pose.position.y = self.status_msg.position.y
        ego_marker.pose.position.z = 0.0
        
        self.ego_marker_pub.publish(ego_marker)



if __name__ == '__main__':

    try:
        pure_pursuit_= PurePursuit()
    except rospy.ROSInterruptException:
        pass