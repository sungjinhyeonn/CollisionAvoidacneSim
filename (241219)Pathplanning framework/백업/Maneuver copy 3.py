import math
import matplotlib.pyplot as plt

class PIDManeuverTest:
    def __init__(self, dt=0.1, max_speed=1.0, max_yaw_rate=math.radians(30)):
        self.dt = dt  # 시간 간격
        self.max_speed = max_speed  # 최대 속도
        self.max_yaw_rate = max_yaw_rate  # 최대 각속도

        # 초기 상태
        self.current_position_x = 0.0
        self.current_position_y = 0.0
        self.current_yaw = 0.0
        self.target_x = 0.0
        self.target_y = 0.0

        # PID 상태 변수
        self.angle_error_sum = 0.0
        self.angle_prev_error = 0.0
        self.distance_error_sum = 0.0
        self.distance_prev_error = 0.0

        # PID 제어 상수
        self.Kp_angle = 0.5
        self.Ki_angle = 0.1
        self.Kd_angle = 0.05
        self.Kp_distance = 0.1
        self.Ki_distance = 0.1
        self.Kd_distance = 0.05

    def set_target(self, target_x, target_y):
        self.target_x = target_x
        self.target_y = target_y

    def pid_control(self):
        """ PID 제어 로직 """
        delta_x = self.target_x - self.current_position_x
        delta_y = self.target_y - self.current_position_y

        # 목표까지의 거리와 각도 계산
        distance = math.sqrt(delta_x**2 + delta_y**2)
        target_angle = math.atan2(delta_y, delta_x)

        # 각도 차이 계산 및 정규화
        angle_diff = target_angle - self.current_yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # 각도 PID 제어
        self.angle_error_sum += angle_diff * self.dt
        angle_derivative = (angle_diff - self.angle_prev_error) / self.dt
        angular_velocity = (self.Kp_angle * angle_diff +
                            self.Ki_angle * self.angle_error_sum +
                            self.Kd_angle * angle_derivative)
        self.angle_prev_error = angle_diff

        # 거리 PID 제어
        self.distance_error_sum += distance * self.dt
        distance_derivative = (distance - self.distance_prev_error) / self.dt
        linear_velocity = (self.Kp_distance * distance +
                           self.Ki_distance * self.distance_error_sum +
                           self.Kd_distance * distance_derivative)
        self.distance_prev_error = distance

        # 속도 제한 적용
        linear_velocity = min(linear_velocity, self.max_speed)
        angular_velocity = max(min(angular_velocity, self.max_yaw_rate), -self.max_yaw_rate)

        # 상태 업데이트
        self.current_yaw += angular_velocity * self.dt
        self.current_position_x += linear_velocity * math.cos(self.current_yaw) * self.dt
        self.current_position_y += linear_velocity * math.sin(self.current_yaw) * self.dt

        return distance

    def simulate(self, steps=100):
        """ 시뮬레이션 실행 """
        x_history = []
        y_history = []
        for _ in range(steps):
            distance_to_target = self.pid_control()

            # 궤적 기록
            x_history.append(self.current_position_x)
            y_history.append(self.current_position_y)

            # 목표에 도달하면 중지
            if distance_to_target < 0.1:
                print(f"Reached target at ({self.current_position_x:.2f}, {self.current_position_y:.2f})")
                break

        return x_history, y_history


# 테스트 실행
if __name__ == "__main__":
    maneuver_test = PIDManeuverTest()

    # 목표 경로점 설정
    maneuver_test.set_target(15, 5)

    # 시뮬레이션 실행
    x_history, y_history = maneuver_test.simulate(steps=200)

    # 결과 플로팅
    plt.figure(figsize=(8, 8))
    plt.plot(x_history, y_history, label="Trajectory")
    plt.scatter([15], [5], color="red", label="Target")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("PID Maneuver Test Trajectory")
    plt.legend()
    plt.grid()
    plt.show()
