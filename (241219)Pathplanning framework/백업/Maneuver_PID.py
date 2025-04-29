from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgCurPose import MsgCurrentPose
import math
import numpy as np
import re

def extract_numbers(input_string):
    """ 정규 표현식을 사용하여 문자열에서 숫자만 추출 """
    numbers = re.findall(r'\d+', input_string)
    return ''.join(numbers)

class PIDController:
    """ PID 제어기 구현 """
    def __init__(self, kp, ki, kd, max_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.integral = 0
        self.last_error = 0

    def control(self, error, dt):
        """ PID 계산 """
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return max(min(output, self.max_output), -self.max_output)

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
        self.curPose = starts[self.id]  # 초기값을 필요에 따라 설정
        self.linear_pid = PIDController(kp=0.01, ki=0.01, kd=0.1, max_output=objConfiguration.getConfiguration('max_speed'))
        self.angular_pid = PIDController(kp=0.1, ki=0.001, kd=0.1, max_output=objConfiguration.getConfiguration('max_yaw_rate'))

    def motion_to_target(self, current_position, target_position, current_yaw, dt):
        """ 목표 위치로 이동을 위한 모션 계산 """
        dx = target_position[0] - current_position[0]
        dy = target_position[1] - current_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        target_yaw = math.atan2(dy, dx)

        angular_velocity = self.angular_pid.control(target_yaw - current_yaw, dt)
        linear_velocity = self.linear_pid.control(distance, dt)

        new_yaw = current_yaw + angular_velocity * dt
        new_x = current_position[0] + linear_velocity * math.cos(new_yaw) * dt
        new_y = current_position[1] + linear_velocity * math.sin(new_yaw) * dt

        return [new_x, new_y, new_yaw]

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
            current_position = [self.getStateValue('current_position_x'), self.getStateValue('current_position_y')]
            target_position = [self.getStateValue('target_x'), self.getStateValue('target_y')]
            print(f'target_position:{target_position}')
            current_yaw = self.getStateValue('current_position_yaw')
            dt = self.getStateValue('dt')

            new_pos = self.motion_to_target(current_position, target_position, current_yaw, dt)
            self.setStateValue('current_position_x', new_pos[0])
            self.setStateValue('current_position_y', new_pos[1])
            self.setStateValue('current_position_yaw', new_pos[2])

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
