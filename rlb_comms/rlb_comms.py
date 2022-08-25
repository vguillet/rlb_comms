

##################################################################################################################
"""
"""

# Built-in/Generic Imports
import json
import os
import queue
import math
from functools import partial
import json

import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import JointState, LaserScan
from rlb_utils.msg import Goal, TeamComm, CommsState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import pandas as pd
import math

from rlb_tools.Caylus_map_loader import load_maps
from rlb_tools.Raster_ray_tracing import check_comms_available
from rlb_config.simulation_parameters import *

##################################################################################################################


class RLB_comms(Node):
    def __init__(self):
        super().__init__('Comms_sim')

        # -> Setup storage dicts
        self.team_members = {}
        self.comm_rays = {}

        # -> Load obstacle_grids
        obstacle_grids = load_maps(
            hard_obstacles=True,
            dense_vegetation=True,
            light_vegetation=True,
            paths=False
        )

        # -> Generate signal blocking probability grid
        self.signal_blocking_prob_grid = \
                obstacle_grids["hard_obstacles"] * hard_obstacles_signal_blocking_prob \
                + obstacle_grids["dense_vegetation"] * dense_vegetation_signal_blocking_prob \
                + obstacle_grids["light_vegetation"] * light_vegetation_signal_blocking_prob

        # -> Clip min/max values
        self.signal_blocking_prob_grid = np.clip(self.signal_blocking_prob_grid, 0, 1)

        # ----------------------------------- Team communications subscriber
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_ALL,
            )

        self.team_comms_subscriber = self.create_subscription(
            msg_type=TeamComm,
            topic="/team_comms",
            callback=self.team_msg_subscriber_callback,
            qos_profile=qos
            )
        
        # ----------------------------------- Comm matrix publisher
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
            )

        self.comms_state_matrix_pub = self.create_publisher(
            msg_type=CommsState,
            topic="/comms_state_matrix",
            qos_profile=qos
        )

        # ----------------------------------- Comms state publishers timer
        self.comms_state_pub_timer = self.create_timer(
            timer_period_sec=0.000001,
            callback=self.publish_comms_states
        )

    def publish_comms_states(self):
        # -> Update comms states
        self.check_comms_state()

        # -> Publish comms states
        agents = []

        for agent_pair in self.agent_pairs:
            agents.append(agent_pair[0])
            agents.append(agent_pair[1])

        # -> Construct communication matrix
        comms_matrix =  pd.DataFrame( 
            index=set(agents),
            columns=set(agents))

        comms_integrities_matrix =  pd.DataFrame(
            index=set(agents),
            columns=set(agents))

        for agent_pair in self.agent_pairs:
            # -> Fill in matrix booleans
            comms_matrix[agent_pair[0]][agent_pair[1]] = self.comm_rays[agent_pair]["comm_state"]
            comms_matrix[agent_pair[1]][agent_pair[0]] = self.comm_rays[agent_pair]["comm_state"]

            # -> Store integrity profiles
            comms_integrities_matrix[agent_pair[0]][agent_pair[1]] = self.comm_rays[agent_pair]["comms_integrity_profile"]
            comms_integrities_matrix[agent_pair[1]][agent_pair[0]] = self.comm_rays[agent_pair]["comms_integrity_profile"]

        msg = CommsState()
        msg.comms_states = json.dumps(
            {
                "comms_state_matrix": comms_matrix.to_json(),
                "comms_integrity_matrix": comms_integrities_matrix.to_json(),
                }
        )

        self.comms_state_matrix_pub.publish(msg)

        for agent in self.team_members.keys():
            msg = CommsState()
            msg.robot_id = agent

            comms_state = {}   

            for agent_pair in self.agent_pairs:
                if agent in agent_pair:
                    other_agent = agent_pair[0] if agent_pair[0] != agent else agent_pair[1]
                    comms_state[other_agent] = {
                        "comm_state": self.comm_rays[agent_pair]["comm_state"],
                        "comms_integrity_profile": self.comm_rays[agent_pair]["comms_integrity_profile"]
                    }

            msg.comms_states = json.dumps(comms_state)

            # -> Publish msg
            self.team_members[agent]["comm_state_publisher"].publish(msg)

    def convert_coords_room_to_pixel(self, point_room):
        from rlb_config.simulation_parameters import images_shape
        from rlb_config.room_paramters import room_x_range, room_y_range

        # -> Calculating differences
        dx_img = images_shape[0]
        dy_img = images_shape[1]

        dx_canvas = abs(room_x_range[0]) + abs(room_x_range[1])
        dy_canvas = abs(room_y_range[0]) + abs(room_y_range[1])

        aspect_ratio = self.signal_blocking_prob_grid.shape[1]/self.signal_blocking_prob_grid.shape[0]
        
        if aspect_ratio < 1:
            dx_canvas *= aspect_ratio
        else:
            dy_canvas *= aspect_ratio

        # -> Solving for scaling factor
        dx_img_shift = dx_img/dx_canvas
        dy_img_shift = dy_img/dy_canvas

        return (int(point_room[0] * dx_img_shift + dx_img/2), int(point_room[1] * dy_img_shift + dy_img/2))
    
    @property
    def agent_pairs(self):
        agent_list = list(self.team_members.keys())
        agent_pairs = [(a, b) for idx, a in enumerate(agent_list) for b in agent_list[idx + 1:]]

        return agent_pairs

    def team_msg_subscriber_callback(self, msg):
        if msg.source not in self.team_members.keys() and msg.source_type == "robot":
            self.add_robot(msg=msg)
            print(f"-> Found {msg.source}")

    def check_comms_state(self):
        try:
            # -> Check if comms are available for every agent tracked
            for agent_pair in self.agent_pairs:
                if pose_tracked == "room":
                    key = "pose"

                elif pose_tracked == "projected":
                    key = "pose_projected"

                x1 = round(self.team_members[agent_pair[0]][key]["x"], 3)
                y1 = round(self.team_members[agent_pair[0]][key]["y"], 3)

                x2 = round(self.team_members[agent_pair[1]][key]["x"], 3)
                y2 = round(self.team_members[agent_pair[1]][key]["y"], 3)

                if isinstance(x1, float) and isinstance(y1, float) and isinstance(x2, float) and isinstance(y1, float): 
                    # -> Check comms state
                    point_1_pix = self.convert_coords_room_to_pixel(point_room=(x1, y1))
                    point_2_pix = self.convert_coords_room_to_pixel(point_room=(x2, y2))

                    self.comm_rays[agent_pair]["comm_state"], self.comm_rays[agent_pair]["comms_integrity_profile"], ray_coordinates = check_comms_available(
                        pose_1=point_1_pix,
                        pose_2=point_2_pix,
                        obstacle_probabilities_grid=self.signal_blocking_prob_grid)
        except:
            pass

    def pose_subscriber_callback(self, robot_id, msg):
        # -> Update position
        self.team_members[robot_id]["pose"]["x"] = msg.pose.position.x
        self.team_members[robot_id]["pose"]["y"] = msg.pose.position.y
        self.team_members[robot_id]["pose"]["z"] = msg.pose.position.z

        # -> Update orientation
        u, v, w = self.__euler_from_quaternion(quat=msg.pose.orientation)

        self.team_members[robot_id]["pose"]["u"] = u
        self.team_members[robot_id]["pose"]["v"] = v
        self.team_members[robot_id]["pose"]["w"] = w

    def pose_projected_subscriber_callback(self, robot_id, msg):
        # -> Update position
        self.team_members[robot_id]["pose_projected"]["x"] = msg.pose.position.x
        self.team_members[robot_id]["pose_projected"]["y"] = msg.pose.position.y
        self.team_members[robot_id]["pose_projected"]["z"] = msg.pose.position.z

        # -> Update orientation
        u, v, w = self.__euler_from_quaternion(quat=msg.pose.orientation)

        self.team_members[robot_id]["pose_projected"]["u"] = u
        self.team_members[robot_id]["pose_projected"]["v"] = v
        self.team_members[robot_id]["pose_projected"]["w"] = w

    def add_robot(self, msg):
        # ---------------- Add team member entry to team members dict
        self.team_members[msg.source] = {
            # -> Pose setup
            "pose_subscriber": None,
            "pose": {
                "x": 0.,
                "y": 0.,
                "z": 0.,
                "u": 0.,
                "v": 0.,
                "w": 0.
                },

            # -> Pose projected setup
            "pose_projected_subscriber": None,
            "pose_projected": {
                "x": 0.,
                "y": 0.,
                "z": 0.,
                "u": 0.,
                "v": 0.,
                "w": 0.
                },
            
            # -> Comms setup
            "comm_state_publisher": None
        }

        # -> Create comm state publisher
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
            )

        self.team_members[msg.source]["comm_state_publisher"] = self.create_publisher(
            msg_type=CommsState,
            topic=f"/{msg.source}/sim/comms_state",
            qos_profile=qos
        )

        # ---------------- Add team member entry to comm_rays
        for agent_pair in self.agent_pairs:
            if agent_pair not in self.comm_rays.keys():
                self.comm_rays[agent_pair] = {
                    "comm_state": True,
                    "comms_integrity_profile": [],
                    }

        # -> Create pose subscribers
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
            )

        self.team_members[msg.source]["pose_subscriber"] = self.create_subscription(
            msg_type=PoseStamped,
            topic=f"/{msg.source}/state/pose",
            callback=partial(self.pose_subscriber_callback, msg.source),
            qos_profile=qos
            )

        # -> Create pose projected subscribers
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
            )

        self.team_members[msg.source]["pose_projected_subscriber"] = self.create_subscription(
            msg_type=PoseStamped,
            topic=f"/{msg.source}/state/pose_projected",
            callback=partial(self.pose_projected_subscriber_callback, msg.source),
            qos_profile=qos
            )

    @staticmethod
    def __euler_from_quaternion(quat):
        """
        Convert quaternion (w in last place) to euler roll, pitch, yaw (rad).
        quat = [x, y, z, w]

        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp) * 180 / math.pi

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp) * 180 / math.pi

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp) * 180 / math.pi

        return roll, pitch, yaw


def main(args=None):
    # `rclpy` library is initialized
    rclpy.init(args=args)

    path_sequence = RLB_comms()

    rclpy.spin(path_sequence)

    path_sequence.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()