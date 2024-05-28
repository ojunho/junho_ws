#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy 
from morai_msgs.msg import GetTrafficLightStatus, CtrlCmd, GPSMessage
from std_msgs.msg import Float64, Int64, Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Imu
from geometry_msgs.msg import  Vector3
from move_base_msgs.msg import MoveBaseActionResult
from cv_bridge import CvBridge
from pyproj import Proj, transform
from nav_msgs.msg import Odometry
from math import pi, sqrt, atan2, radians
from lidar_object_detection.msg import ObjectInfo
from slidewindow import SlideWindow
from visualization_msgs.msg import MarkerArray

import tf
import time
import cv2
import numpy as np

class Obstacle:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class PID():
  def __init__(self,kp,ki,kd):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.p_error = 0.0
    self.i_error = 0.0
    self.d_error = 0.0

  def pid_control(self, cte):
    self.d_error = cte-self.p_error
    self.p_error = cte
    self.i_error += cte

    return self.kp*self.p_error + self.ki*self.i_error + self.kd*self.d_error

class  LaneDetection:
    def __init__(self):
        rospy.init_node("lane_detection_node")

        self.ctrl_cmd_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        self.stopline_flag_pub = rospy.Publisher('/stopline_flag', Bool, queue_size=1)

        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.camCB)
        rospy.Subscriber("/bounding_box", MarkerArray, self.objectCB)
        
        #-------------------------------------------------------------------------------------- #
        self.proj_UTM = Proj(proj='utm', zone = 52, elips='WGS84', preserve_units=False)
        self.bridge = CvBridge()
        self.ctrl_cmd_msg = CtrlCmd()
        # self.status_msg   = EgoStatus()
        self.traffic_msg = GetTrafficLightStatus()

        self.ctrl_cmd_msg.longlCmdType = 2
        self.is_gps = True
        self.is_imu = True

        self.slidewindow = SlideWindow()

        self.traffic_flag = 0
        self.prev_signal = 0
        self.signal = 0
        self.stopline_flag = 0
        self.img = []
        self.warped_img = []
        self.grayed_img = []
        self.out_img = []
        self.yellow_img = []
        self.white_img = []
        self.img_hsv = []
        self.h = []
        self.s = []
        self.v = []
        self.bin_img = []
        self.left_indices = []
        self.right_indices = []
        self.lidar_obstacle_info = []
        self.gt_heading = 0
        
        self.heading =0
        self.x_location = 480
        self.last_x_location = 480

        self.prev_center_index = 480
        self.center_index = 480
        self.standard_line = 480
        self.degree_per_pixel = 0
        self.avoid_x_location = 480

        self.current_lane = 2
        self.avoid_status = 'lanekeeping'
        self.sliding_window_select_line = 'Right'
        
        self.current_waypoint = 0

        self.up_hist_end_line = 400
        self.down_hist_start_line = 400

        self.fusion_result = []


        #------------------------------------- 동적 장애물 파라미터 -------------------------------------#
        self.obstacle_list = []

        self.obstacle_flag = False
        self.brake_cnt = 0

        #-----------------------------------------------------------------------------------------------#
        rate = rospy.Rate(20)  # hz 
        while not rospy.is_shutdown():

            if len(self.img)!= 0:

                y, x = self.img.shape[0:2]
                
                self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                
                h, s, v = cv2.split(self.img_hsv)
                
                # print(type(h))

                # 터널 밖 노란 선 hsv
                yellow_lower_1 = np.array([15, 75, 180])
                yellow_upper_1 = np.array([23, 150, 245])
                yellow_range_1 = cv2.inRange(self.img_hsv, yellow_lower_1, yellow_upper_1)

                # 터널 안 노란 선 hsv
                yellow_lower_2 = np.array([24, 110, 50])
                yellow_upper_2 = np.array([90, 180, 85])
                yellow_range_2 = cv2.inRange(self.img_hsv, yellow_lower_2, yellow_upper_2)
                
                # 터널 밖 흰 선 hsv
                white_lower_bound1 = np.array([10, 27, 210])
                white_upper_bound1 = np.array([25, 45, 255])
                white_mask1 = cv2.inRange(self.img_hsv, white_lower_bound1, white_upper_bound1)
                
                # 터널 안 흰 선 hsv
                white_lower_bound2 = np.array([90, 40, 60])
                white_upper_bound2 = np.array([110, 70, 90])
                white_mask2 = cv2.inRange(self.img_hsv, white_lower_bound2, white_upper_bound2)

                self.yellow_range = cv2.bitwise_or(yellow_range_1, yellow_range_2)

                self.white_range = cv2.bitwise_or(white_mask1, white_mask2)
                
                combined_range = cv2.bitwise_or(self.yellow_range, self.white_range)
                filtered_img = cv2.bitwise_and(self.img, self.img, mask=combined_range)


                left_margin = 330
                top_margin = 322
                src_point1 = [0, 444]      # 왼쪽 아래
                src_point2 = [left_margin, top_margin]
                src_point3 = [x-left_margin, top_margin]
                src_point4 = [x , 444]  

                src_points = np.float32([src_point1,src_point2,src_point3,src_point4])
                

                dst_point1 = [x//4, 540]    # 왼쪽 아래
                dst_point2 = [x//4, 0]      # 왼쪽 위
                dst_point3 = [x//4*3, 0]    # 으론쪽 위
                dst_point4 = [x//4*3, 540]  # 오른쪽 아래

                dst_points = np.float32([dst_point1,dst_point2,dst_point3,dst_point4])
                
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                self.warped_img = cv2.warpPerspective(filtered_img, matrix, [x,y])

                
                self.grayed_img = cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2GRAY)
                
                # 이미지 이진화
                self.bin_img = np.zeros_like(self.grayed_img)

                self.bin_img[self.grayed_img > 60] = 1
                # self.bin_img[self.grayed_img > 150] = 1 #  > 150
                
                histogram_y = np.sum(self.bin_img, axis=1)
                # print(f"histogram_y: {histogram_y}")            
                if self.x_location == None :
                    self.x_location = self.last_x_location
                else :
                    self.last_x_location = self.x_location

                    
                
                # #--------------------------------------------- 장애물 ---------------------------------------------#
                #     if (0.4 < self.obstacle_dynamic_static_x < 1.5) and self.finish_detection == False:
                #         self.publishMotorSteerMsg(0, self.steer_msg)
                #         self.obstacle_y_list.append(self.obstacle_dynamic_static_y)
                        
                #         if len(self.obstacle_y_list) == 400 and self.finish_detection == False :
                #             self.finish_detection = True
                
                #             if abs(self.obstacle_y_list[100] - self.obstacle_y_list[-3]) > 0.08:
                #                 self.is_dynamic= True
                #                 self.is_static = False
                #             else:
                #                 self.is_static = True
                #                 self.is_dynamic = False

                #         continue

                #     if self.is_dynamic:
                #         if -0.4 < self.obstacle_dynamic_static_y < 0.5 and not (self.obstacle_dynamic_static_x == 0 and self.obstacle_dynamic_static_y == 0):
                #             self.publishMotorSteerMsg(0, self.steer_msg)
                #             self.obstacle_y_list =[]
                #             continue

                #         elif self.obstacle_dynamic_static_x == 0 and self.obstacle_dynamic_static_y == 0:
                #             self.is_dynamic = False
                #             self.obstacle_y_list = []
                #             self.finish_detection = False

                #     elif self.is_static:
                #         if self.finish_detection == True:
                #             self.is_goal_arrived = False
                #             self.is_static = False
                #             self.obstacle_y_list = []
                #             continue
                    
                #     # 장애물을 지나갔다는 판단
                #     if self.obstacle_dynamic_static_x < 0.1 and not (self.obstacle_dynamic_static_x == 0 and self.obstacle_dynamic_static_y == 0):
                #         self.finish_detection = False
                #         self.finish_second_detection = False
                #         self.obstacle_y_list = []
                #         self.dynamic_obstacle_y_list = []
                #         self.is_dynamic = False
                #         self.is_static = False
                #     #--------------------------------------------------------------------------------------------------#

                self.out_img, self.x_location, _ = self.slidewindow.slidewindow(self.bin_img)
                pid = PID(0.015, 0.003, 0.010)

                if self.x_location == None :
                    self.x_location = self.last_x_location
                else :
                    self.last_x_location = self.x_location
                
                

                self.standard_line = x//2
                self.degree_per_pixel = 1/x
                self.prev_center_index = self.center_index
                self.center_index = self.x_location

                motor_msg = 25

                angle = pid.pid_control(self.center_index - 480) 
                servo_msg = -radians(angle)
                        

                # 장애물 코드 
                if len(self.obstacle_list) > 0:
                    for obstacle in self.obstacle_list:
                        distance = (obstacle.x ** 2 + obstacle.y ** 2)**0.5
                        if distance < 11:
                            self.obstacle_flag = True
                        else:
                            self.obstacle_flag = False
                else:
                    self.obstacle_flag = False
                    

                if self.obstacle_flag == True:
                    self.brake_cnt += 1
                    self.publishCtrlCmd(0, servo_msg, 1)
                    continue
                
























                self.publishCtrlCmd(motor_msg, servo_msg, 0)


                

                # print("self.x_location", self.x_location)


                # cv2.imshow("self.img", self.img)
                # cv2.imshow("self.img_hsv", self.img_hsv)
                # cv2.imshow("h", h)
                # cv2.imshow("s", s)
                # cv2.imshow("v", v)

                cv2.imshow("filtered_img", filtered_img)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                # cv2.imshow("self.warped_img", self.warped_img)

                # cv2.imshow("out_img", self.out_img)
                # cv2.imshow("grayed_img", self.grayed_img)
                # cv2.imshow("self.bin_img", self.bin_img)
                cv2.waitKey(1)

            rate.sleep()

    def camCB(self, msg):
        self.img = self.bridge.compressed_imgmsg_to_cv2(msg)

    def objectCB(self, msg):
        self.obstacle_list = []
        for marker in msg.markers:
            obstacle = Obstacle(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
            self.obstacle_list.append(obstacle)

    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        self.ctrl_cmd_msg.velocity = motor_msg
        self.ctrl_cmd_msg.steering = servo_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)


if __name__ == "__main__":
    try: 
        lane_detection_node = LaneDetection()
    except rospy.ROSInterruptException:
        pass
