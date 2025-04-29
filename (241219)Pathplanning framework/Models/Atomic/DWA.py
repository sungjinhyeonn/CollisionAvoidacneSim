from enum import Enum
import numpy as np
import math
from shapely.geometry import Polygon, Point

class RobotType(Enum):
    circle = 0
    rectangle = 1

class DWAPlanner:
    def __init__(self, config):
        # Configuration parameters
        self.config = config
        self.robot_type = RobotType.rectangle
        self.robot_length = 3
        self.robot_width = 3
        self.robot_radius = config.getConfiguration('robot_radius')

    def calc_dwa(self, current_pose, goal, obstacles, terrain_polygons):
        """
        DWA 기반 지역 경로 계획 수행
        Args:
            current_pose: 현재 위치 [x, y, yaw, v, omega]
            goal: 목표 위치 (x, y)
            obstacles: 장애물 정보
            terrain_polygons: 지형 정보
        Returns:
            target_state: 다음 목표 상태 [x, y, yaw]
        """
        # Dynamic Window 생성
        dw = self.calc_dynamic_window(current_pose)
        
        # 최적의 제어 입력과 경로를 선택
        best_u, best_trajectory = self.calc_control_and_trajectory(
            current_pose, 
            dw, 
            goal, 
            obstacles, 
            terrain_polygons
        )
        
        # 최적 경로의 마지막 상태 반환 (x, y, yaw만)
        target_state = best_trajectory[-1].tolist()[:3]
        return target_state

    def plan(self, current_pose, goal, obstacles, terrain_polygons):
        """
        내부적으로 사용되는 계획 메서드
        """
        dw = self.calc_dynamic_window(current_pose)
        best_u, best_trajectory = self.calc_control_and_trajectory(
            current_pose, dw, goal, obstacles, terrain_polygons
        )
        return best_u, best_trajectory

    def calc_dynamic_window(self, x):
        Vs = [self.config.getConfiguration('min_speed'),
              self.config.getConfiguration('max_speed'),
              -self.config.getConfiguration('max_yaw_rate'),
              self.config.getConfiguration('max_yaw_rate')]

        Vd = [x[3] - self.config.getConfiguration('max_accel') * self.config.getConfiguration('dt'),
              x[3] + self.config.getConfiguration('max_accel') * self.config.getConfiguration('dt'),
              x[4] - self.config.getConfiguration('max_delta_yaw_rate') * self.config.getConfiguration('dt'),
              x[4] + self.config.getConfiguration('max_delta_yaw_rate') * self.config.getConfiguration('dt')]

        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw

    def calc_control_and_trajectory(self, x, dw, goal, ob, terrain_polygons):
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        for v in np.arange(dw[0], dw[1], self.config.getConfiguration('v_resolution')):
            for y in np.arange(dw[2], dw[3], self.config.getConfiguration('yaw_rate_resolution')):
                trajectory = self.predict_trajectory(x_init, v, y)
                
                to_goal_cost = self.config.getConfiguration('to_goal_cost_gain') * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.config.getConfiguration('speed_cost_gain') * (self.config.getConfiguration('max_speed') - trajectory[-1, 3])
                ob_cost = self.config.getConfiguration('obstacle_cost_gain') * self.calc_obstacle_cost(trajectory, ob)
                terrain_cost = self.calc_terrain_cost(trajectory, terrain_polygons)

                final_cost = to_goal_cost + speed_cost + ob_cost + terrain_cost

                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory

        return best_u, best_trajectory

    def predict_trajectory(self, x_init, v, y):
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.config.getConfiguration('predict_time'):
            x = self.motion(x, [v, y], self.config.getConfiguration('dt'))
            trajectory = np.vstack((trajectory, x))
            time += self.config.getConfiguration('dt')
        return trajectory

    def motion(self, x, u, dt):
        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def calc_terrain_cost(self, trajectory, terrain_polygons):
        cost = 0.0
        for polygon_points in terrain_polygons:
            polygon = Polygon(polygon_points)
            for point in trajectory[:, :2]:
                p = Point(point[0], point[1])
                if polygon.contains(p):
                    return float("Inf")
        return cost

    def calc_to_goal_cost(self, trajectory, goal):
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        current_heading = trajectory[-1, 2]
        cost_angle = error_angle - current_heading
        cost_angle = (cost_angle + math.pi) % (2 * math.pi) - math.pi
        cost = min(abs(cost_angle), 2 * math.pi - abs(cost_angle))
        return cost

    def calc_obstacle_cost(self, trajectory, ob):
        if not isinstance(ob, np.ndarray):
            ob = np.array(ob) if isinstance(ob, list) else np.array(list(ob.values()))

        if ob is None or len(ob) == 0:
            return 0

        ox = ob[:, 0]
        oy = ob[:, 1]
        dx = trajectory[:, 0][:, np.newaxis] - ox[np.newaxis, :]
        dy = trajectory[:, 1][:, np.newaxis] - oy[np.newaxis, :]
        r = np.hypot(dx, dy)

        if self.robot_type == RobotType.rectangle:
            yaw = trajectory[:, 2]
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rot = np.transpose(rot, [2, 0, 1])
            local_ob = ob[:, None] - trajectory[:, 0:2]
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            local_ob = np.array([local_ob @ x for x in rot])
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            upper_check = local_ob[:, 0] <= self.robot_length / 2
            right_check = local_ob[:, 1] <= self.robot_width / 2
            bottom_check = local_ob[:, 0] >= -self.robot_length / 2
            left_check = local_ob[:, 1] >= -self.robot_width / 2
            if (np.logical_and(np.logical_and(upper_check, right_check),
                             np.logical_and(bottom_check, left_check))).any():
                return float("Inf")

        elif self.robot_type == RobotType.circle:
            if np.array(r <= self.robot_radius).any():
                return float("Inf")

        min_r = np.min(r)
        return 1.0 / min_r if min_r != 0 else float("Inf") 