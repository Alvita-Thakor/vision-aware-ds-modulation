import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
import numpy as np
from math import atan2, cos, sin
sys.path.append('/mnt/e/PROJECTS/vision-aware-ds-modulation')
from src.estimation.kalman_filter import Kalman_filter
from src.controller.controller import Controller
from src.entities.position import Position
from src.entities.obstacle import Obstacle
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped

def pixel_to_ground(u, v):
    fx = 277.0
    fy = 277.0
    cx = 160.0
    cy = 120.0

    # Ray in the camera optical frame
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0

    # Camera optical frame → robot base frame
    ray_x = z
    ray_y = -x
    ray_z = -y

    camera_x = 0.043
    camera_y = 0.0
    camera_z = 0.119

    # Ground plane: robot z = 0
    if abs(ray_z) < 1e-6:
        return None
    t = -camera_z / ray_z

    ground_x = camera_x + t * ray_x
    ground_y = camera_y + t * ray_y

    return ground_x, ground_y

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()
        self.kalman = Kalman_filter()
        self.controller = Controller()
        self.goal = Position(3.0, 0.0)
        self.obstacle_position = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.cmd_vel_publisher = self.create_publisher(TwistStamped,'/cmd_vel',10)
        self.start_time = self.get_clock().now()
        self.control_timer = self.create_timer(0.1,self.control_callback)
        self.odom_subscription = self.create_subscription(Odometry,'/odom',self.odom_callback,10)
        self.subscription = self.create_subscription(Image,'/camera/image_raw',self.image_callback,10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        height, width = frame.shape[:2]
        image_center_x = width // 2
        best_contour = None
        best_distance = float('inf')

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)

                contour_center_x = x + w // 2
                distance = abs(contour_center_x - image_center_x)

                if distance < best_distance:
                    best_distance = distance
                    best_contour = (x, y, w, h)

        if best_contour is not None:
            x, y, w, h = best_contour
            obstacle_x = x + w // 2
            obstacle_y = y + h
            self.get_logger().info(
    f"Obstacle pixel: ({obstacle_x:.1f}, {obstacle_y:.1f})"
)

            ground_point = pixel_to_ground(obstacle_x, obstacle_y)

            if ground_point is None:
                return

            ground_x, ground_y = ground_point
            measurement = (ground_x, ground_y)

            if not self.kalman.initialized:
                self.kalman.initialize(measurement)
            else:
                self.kalman.predict()
                self.kalman.update(measurement)

            filtered_x, filtered_y = self.kalman.get_position()

            world_x = (self.robot_x + cos(self.robot_yaw) * filtered_x - sin(self.robot_yaw) * filtered_y)
            world_y = (self.robot_y + sin(self.robot_yaw) * filtered_x + cos(self.robot_yaw) * filtered_y)

            self.obstacle_position = Position(world_x, world_y)
            #self.obstacle_position = Position(1.5, 0.0)
            self.get_logger().info(f'Robot: ({self.robot_x:.2f}, {self.robot_y:.2f}) | 'f'World obstacle: ({world_x:.2f}, {world_y:.2f})')
            self.get_logger().info(f'Raw: ({ground_x:.2f}, {ground_y:.2f}) | 'f'Filtered: ({filtered_x:.2f}, {filtered_y:.2f})')

            cv2.circle(frame,(obstacle_x, obstacle_y),5,(0, 0, 255),-1)
            cv2.rectangle(frame,(x, y),(x + w, y + h),(0, 255, 0),2)

        cv2.imshow('TurtleBot Camera', frame)
        cv2.waitKey(1)

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
        cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        self.robot_yaw = atan2(sin_yaw, cos_yaw)

    def control_callback(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if elapsed < 5.0:
            return
        if self.obstacle_position is None:
            return

        robot_position = Position(
            self.robot_x,
            self.robot_y
        )

        obstacle = Obstacle(
            position=self.obstacle_position,
            radius=0.5
        )

        velocity = self.controller.compute_velocity(robot_position,self.goal,[obstacle],0.2)

        speed = (velocity.vx**2 + velocity.vy**2) ** 0.5
        desired_yaw = atan2(velocity.vy, velocity.vx)
        heading_error = atan2(sin(desired_yaw - self.robot_yaw),cos(desired_yaw - self.robot_yaw))

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear.x = min(0.10, speed) * max(0.0, cos(heading_error))
        cmd.twist.angular.z = max(-1.0, min(1.0, 2.0 * heading_error))

        self.cmd_vel_publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = CameraSubscriber()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()