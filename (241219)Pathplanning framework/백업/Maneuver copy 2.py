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
        self.addInputPort("StopSimulation")
        self.addOutputPort("ManeuverState")

        self.id = int(extract_numbers(self.ID))
        starts = objConfiguration.getConfiguration('agent_starts')

        self.dt = objConfiguration.getConfiguration('dt')
        self.addStateVariable('dt', self.dt)
        self.addStateVariable("mode", "WAIT")
        self.addStateVariable('current_position_x', starts[self.id][0])
        self.addStateVariable('current_position_y', starts[self.id][1])
        self.addStateVariable('current_position_yaw', starts[self.id][2])
        self.addStateVariable('current_position_lin_vel', starts[self.id][3])
        self.addStateVariable('current_position_ang_vel', starts[self.id][4])
         # 목표 위치 초기화
        self.addStateVariable('target_position_x', 0)
        self.addStateVariable('target_position_y', 0)
         

    def funcExternalTransition(self, strPort, objEvent):
        """ 외부 이벤트에 대한 처리 """
        if strPort == "StopSimulation":
            self.setStateValue("mode", "DEAD")
            self.continueTimeAdvance()
        elif strPort == "RequestManeuver":
            self.setStateValue("mode", "POSITION")
            self.setStateValue("target_x", objEvent.dblPositionN)
            self.setStateValue("target_y", objEvent.dblPositionE)

    def funcInternalTransition(self):
        """ 내부 상태 전이 로직 """
        if self.getStateValue('mode') == 'POSITION':
            self.pg_control()
            
    def pg_control(self):
        """ Proportional Guidance (PG) 제어 로직 구현 """
        current_x = self.getStateValue('current_position_x')
        current_y = self.getStateValue('current_position_y')
        current_yaw = self.getStateValue('current_position_yaw')
        target_x = self.getStateValue('target_position_x')
        target_y = self.getStateValue('target_position_y')

        # 목표까지의 벡터 계산
        delta_x = target_x - current_x
        delta_y = target_y - current_y
        
        # 목표까지의 거리와 각도
        distance = math.sqrt(delta_x**2 + delta_y**2)
        target_angle = math.atan2(delta_y, delta_x)

        # 현재 각도와 목표 각도 사이의 각도 차이 계산
        angle_diff = target_angle - current_yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # 각도 정규화

        # 비례 제어 계수
        Kp_angle = 0.5  # 각도 조정을 위한 비례 계수
        Kp_distance = 0.1  # 거리 조정을 위한 비례 계수

        # 각속도와 선속도 계산
        angular_velocity = Kp_angle * angle_diff
        linear_velocity = Kp_distance * distance if distance > 1 else 0  # 최소 거리보다 클 때만 속도 적용

        # 속도 제한 적용
        max_linear_velocity = self.objConfiguration.getConfiguration('max_speed')
        max_angular_velocity = self.objConfiguration.getConfiguration('max_yaw_rate')
        linear_velocity = min(linear_velocity, max_linear_velocity)
        angular_velocity = max(min(angular_velocity, max_angular_velocity), -max_angular_velocity)

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
        return 0.1 if self.getStateValue('mode') == 'POSITION' else float('inf')

    def funcSelect(self):
        """ 선택 함수 """
        pass
