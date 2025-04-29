from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgCurPose import MsgCurrentPose
import math
import numpy as np
import re

def extract_numbers(input_string):
    """ 정규 표현식을 사용하여 문자열에서 숫자만 추출 """
    numbers = re.findall(r'\d+', input_string)
    return ''.join(numbers)

class Maneuver(DEVSAtomicModel):
    """ Maneuver 클래스 구현 """
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)
        self.objConfiguration = objConfiguration
        self.addInputPort("RequestManeuver")
        self.addInputPort("StopSimulation_IN")
        self.addOutputPort("ManeuverState")

        self.id = int(extract_numbers(self.ID))
        starts = objConfiguration.getConfiguration('agent_starts')

        self.dt = 0.1
        self.addStateVariable('dt', self.dt)
        self.addStateVariable("mode", "WAIT")
        self.addStateVariable('current_position_x', starts[self.id][0])
        self.addStateVariable('current_position_y', starts[self.id][1])
        self.addStateVariable('current_position_yaw', starts[self.id][2])
        self.addStateVariable('current_position_lin_vel', starts[self.id][3])
        self.addStateVariable('current_position_ang_vel', starts[self.id][4])
        self.addStateVariable('target_x', 0)
        self.addStateVariable('target_y', 0)

        # 차량 동역학 파라미터
        self.wheelbase = 2.0  # 축거
        self.max_steer_angle = 0.6  # 최대 조향각 (rad)
        self.look_ahead_distance = 3.0  # 전방주시거리

    def funcExternalTransition(self, strPort, objEvent):
        """ 외부 이벤트에 대한 처리 """
        if strPort == "StopSimulation_IN":
            print(f"{self.ID}: Received StopSimulation message")
            self.setStateValue("mode", "WAIT")
            self.continueTimeAdvance()
        elif strPort == "RequestManeuver":
            self.setStateValue("mode", "POSITION")
            self.setStateValue("target_x", objEvent.dblPositionE)
            self.setStateValue("target_y", objEvent.dblPositionN)

    def funcInternalTransition(self):
        """ 내부 상태 전이 로직 """
        if self.getStateValue('mode') == 'POSITION':
            self.advanced_control()

    def advanced_control(self):
        """향상된 제어 로직 - 부드러운 움직임"""
        # 현재 상태
        current_x = self.getStateValue('current_position_x')
        current_y = self.getStateValue('current_position_y')
        current_yaw = self.getStateValue('current_position_yaw')
        current_velocity = self.getStateValue('current_position_lin_vel')
        current_angular_vel = self.getStateValue('current_position_ang_vel')
        target_x = self.getStateValue('target_x')
        target_y = self.getStateValue('target_y')

        # 목표까지의 벡터 계산
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)

        # 각도 차이 계산 및 정규화
        angle_diff = target_angle - current_yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # 부드러운 각속도 제어 - 게인 감소
        max_yaw_rate = self.objConfiguration.getConfiguration('max_yaw_rate')
        target_angular_velocity = np.clip(
            1.0 * angle_diff,  # 게인을 2.0에서 1.0으로 감소
            -max_yaw_rate,
            max_yaw_rate
        )

        # 각속도 변화율 제한 - 더 부드럽게
        angular_accel = 0.8  # 1.5에서 0.8로 감소
        angular_vel_diff = target_angular_velocity - current_angular_vel
        angular_vel_diff = np.clip(angular_vel_diff, -angular_accel * self.dt, angular_accel * self.dt)
        angular_velocity = current_angular_vel + angular_vel_diff

        # 속도 제어 - 감속 거리 증가
        max_speed = self.objConfiguration.getConfiguration('max_speed')
        min_speed = self.objConfiguration.getConfiguration('min_speed')
        target_tolerance = self.objConfiguration.getConfiguration('target_tolerance')
        
        # 감속 거리를 더 짧게 설정
        stop_distance = target_tolerance * 2  # 5배에서 2배로 감소
        
        if distance < stop_distance:
            # 거리에 따른 비선형 감속 - 더 늦게 감속 시
            speed_factor = (distance / stop_distance)  # 제곱 제거하여 선형적으로 감속
            target_velocity = max(min_speed, min_speed + (max_speed - min_speed) * speed_factor)
            
            if distance < target_tolerance:  # 목표 지점 도달 시
                target_velocity = 0.0  # 완전 정지
                angular_velocity = 0.0  # 회전도 정지
        else:
            target_velocity = max_speed

        # 회전 시 속도 감소 - 감소 정도를 줄임
        angular_factor = 1.0 - (abs(angular_velocity) / max_yaw_rate)
        angular_factor = max(0.6, angular_factor)  # 0.4에서 0.6으로 수정하여 회전 시 속도 감소를 줄임
        target_velocity *= angular_factor

        # 가속도 제한
        max_accel = self.objConfiguration.getConfiguration('max_accel')
        velocity_diff = target_velocity - current_velocity
        acceleration = np.clip(velocity_diff / self.dt, -max_accel, max_accel)
        
        # 부드러운 가속
        alpha = 0.6
        filtered_accel = acceleration * alpha + (1 - alpha) * self.last_acceleration if hasattr(self, 'last_acceleration') else acceleration
        self.last_acceleration = filtered_accel
        
        linear_velocity = np.clip(current_velocity + filtered_accel * self.dt, 0.0, max_speed)  # 최소 속도를 0으로 설정

        # 상태 업데이트
        new_yaw = current_yaw + angular_velocity * self.dt
        new_x = current_x + linear_velocity * math.cos(new_yaw) * self.dt
        new_y = current_y + linear_velocity * math.sin(new_yaw) * self.dt

        # 상태 변수 업데이트
        self.setStateValue('current_position_x', new_x)
        self.setStateValue('current_position_y', new_y)
        self.setStateValue('current_position_yaw', new_yaw)
        self.setStateValue('current_position_lin_vel', linear_velocity)
        self.setStateValue('current_position_ang_vel', angular_velocity)

    def funcOutput(self):
        """ 출력 생성 """
        if self.getStateValue('mode') == 'POSITION':
            objRequestMessage = MsgCurrentPose(self.ID, 
                                               self.getStateValue('current_position_x'), 
                                               self.getStateValue('current_position_y'),
                                               self.getStateValue('current_position_yaw'),
                                               self.getStateValue('current_position_lin_vel'),
                                               self.getStateValue('current_position_ang_vel'),
                                               self.getTime())
            self.addOutputEvent("ManeuverState", objRequestMessage)

    def funcTimeAdvance(self):
        """ 시간 전이 함수 """
        if self.getStateValue('mode') == "WAIT":
            return float('inf')
        return 0.1

    def funcSelect(self):
        """ 선택 함수 """
        pass
