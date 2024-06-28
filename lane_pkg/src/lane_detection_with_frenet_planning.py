#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy 
from morai_msgs.msg import GetTrafficLightStatus, CtrlCmd, GPSMessage
from std_msgs.msg import Float64, Int64, Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Imu
from geometry_msgs.msg import Vector3
from move_base_msgs.msg import MoveBaseActionResult
from nav_msgs.msg import Odometry
from pyproj import Proj, transform
from math import pi, sqrt, atan2, radians
from lidar_object_detection.msg import ObjectInfo
from slidewindow import SlideWindow
from visualization_msgs.msg import MarkerArray

import tf
import time
import cv2
import numpy as np
from frenet_planning import *

class Obstacle:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def pid_control(self, cte):
        self.d_error = cte - self.p_error
        self.p_error = cte
        self.i_error += cte
        return self.kp * self.p_error + self.ki * self.i_error + self.kd * self.d_error

class LaneDetection:
    def __init__(self):
        rospy.init_node("lane_detection_node")
        self.ctrl_cmd_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        self.stopline_flag_pub = rospy.Publisher('/stopline_flag', Bool, queue_size=1)
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.camCB)
        rospy.Subscriber("/bounding_box", MarkerArray, self.objectCB)

        self.proj_UTM = Proj(proj='utm', zone=52, elips='WGS84', preserve_units=False)
        self.bridge = CvBridge()
        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 2

        self.slidewindow = SlideWindow()

        self.img = []
        self.obstacle_list = []
        self.current_position = (0, 0)

        self.map_center = np.load('map_center.npy')  # 맵 중심 좌표 로드
        self.maps = np.zeros(self.map_center.shape[0])

        for i in range(len(self.map_center)):
            x = self.map_center[i][0]
            y = self.map_center[i][1]
            self.maps[i] = get_frenet(x, y, self.map_center[:, 0], self.map_center[:, 1])[0]

        self.pid = PID(0.015, 0.003, 0.010)

        rate = rospy.Rate(20)  # hz
        while not rospy.is_shutdown():
            if len(self.img) != 0:
                self.process_image()
                self.get_frenet_coords()

                if len(self.obstacle_list) > 0:
                    optimal_path = self.plan_path()
                    self.follow_path(optimal_path)
                else:
                    self.follow_lane()

                cv2.imshow("filtered_img", self.filtered_img)
                cv2.imshow("self.warped_img", self.warped_img)
                cv2.imshow("out_img", self.out_img)
                cv2.imshow("grayed_img", self.grayed_img)
                cv2.imshow("self.bin_img", self.bin_img)
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

    def get_frenet_coords(self):
        mapx = self.map_center[:, 0]
        mapy = self.map_center[:, 1]
        self.s, self.d = get_frenet(self.current_position[0], self.current_position[1], mapx, mapy)
        self.obs_frenet = [get_frenet(ob.x, ob.y, mapx, mapy) for ob in self.obstacle_list]

    def plan_path(self):
        si, si_d, si_dd = self.s, 0, 0  # 현재 s 방향 초기 조건
        sf_d, sf_dd = 1, 0  # 목표 s 방향 조건 (TARGET_SPEED를 1로 설정)
        di, di_d, di_dd = self.d, 0, 0  # 현재 d 방향 초기 조건
        df_d, df_dd = 0, 0  # 목표 d 방향 조건
        opt_d = di  # 초기 d 값

        path, opt_ind = frenet_optimal_planning(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, self.obs_frenet, self.map_center[:, 0], self.map_center[:, 1], self.maps, opt_d)

        return path[opt_ind]

    def follow_path(self, optimal_path):
        for (x, y, yaw) in zip(optimal_path.x, optimal_path.y, optimal_path.yaw):
            self.control_vehicle_to_follow_path(x, y, yaw)

    def control_vehicle_to_follow_path(self, x, y, yaw):
        angle = self.pid.pid_control(self.center_index - 480)
        servo_msg = -radians(angle)
        motor_msg = 25  # 필요에 따라 조정하십시오.
        self.publishCtrlCmd(motor_msg, servo_msg, 0)

    def process_image(self):

        y, x = self.img.shape[0:2]
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        yellow_lower_1 = np.array([15, 75, 180])
        yellow_upper_1 = np.array([23, 150, 245])
        yellow_range_1 = cv2.inRange(self.img_hsv, yellow_lower_1, yellow_upper_1)

        yellow_lower_2 = np.array([24, 100, 35])
        yellow_upper_2 = np.array([90, 250, 50])
        yellow_range_2 = cv2.inRange(self.img_hsv, yellow_lower_2, yellow_upper_2)

        white_lower_bound1 = np.array([10, 27, 210])
        white_upper_bound1 = np.array([25, 45, 255])
        white_mask1 = cv2.inRange(self.img_hsv, white_lower_bound1, white_upper_bound1)

        white_lower_bound2 = np.array([90, 40, 60])
        white_upper_bound2 = np.array([110, 70, 90])
        white_mask2 = cv2.inRange(self.img_hsv, white_lower_bound2, white_upper_bound2)

        self.yellow_range = cv2.bitwise_or(yellow_range_1, yellow_range_2)
        self.white_range = cv2.bitwise_or(white_mask1, white_mask2)

        combined_range = cv2.bitwise_or(self.yellow_range, self.white_range)
        self.filtered_img = cv2.bitwise_and(self.img, self.img, mask=combined_range)

        left_margin = 330
        top_margin = 322

        src_points = np.float32([[0, 444], [left_margin, top_margin], [x - left_margin, top_margin], [x, 444]])
        dst_points = np.float32([[x // 4, 540], [x // 4, 0], [x // 4 * 3, 0], [x // 4 * 3, 540]])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        self.warped_img = cv2.warpPerspective(self.filtered_img, matrix, [x, y])
        self.grayed_img = cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2GRAY)
        self.bin_img = np.zeros_like(self.grayed_img)
        self.bin_img[self.grayed_img > 20] = 1
        self.out_img, self.x_location, _ = self.slidewindow.slidewindow(self.bin_img)
        
        if self.x_location is None:
            self.x_location = self.last_x_location
        else:
            self.last_x_location = self.x_location
        self.center_index = self.x_location

    def follow_lane(self):
        angle = self.pid.pid_control(self.center_index - 480)
        servo_msg = -radians(angle)
        motor_msg = 25  # 필요에 따라 조정하십시오.
        self.publishCtrlCmd(motor_msg, servo_msg, 0)

if __name__ == "__main__":
    try:
        lane_detection_node = LaneDetection()
    except rospy.ROSInterruptException:
        pass
