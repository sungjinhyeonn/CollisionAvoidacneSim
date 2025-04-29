from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgCurPose import MsgCurrentPose
from Models.Message.MsgManeuverState import MsgManeuverState
import math
import numpy as np
import re

def extract_numbers(input_string):
    # 정규 표현식을 사용하여 문자열에서 숫자만 추출
    numbers = re.findall(r'\d+', input_string)
    # 추출된 숫자를 하나의 문자열로 연결하여 반환
    return ''.join(numbers)

class PIDController:
    def __init__(self, kp, ki, kd, max_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.integral = 0
        self.last_error = 0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return max(min(output, self.max_output), -self.max_output)
    
class Maneuver(DEVSAtomicModel):
    def __init__(self, ID, objConfiguration):

        super().__init__(ID)
        self.objConfiguration = objConfiguration

        self.addInputPort("RequestManeuver")
        self.addInputPort("StopSimulation")

        self.addOutputPort("ManeuverState")

        self.id = int(extract_numbers(self.ID))
        starts = objConfiguration.getConfiguration('agent_starts')
        self.isNavi = True


        # 설정 값을 가져와서 속성으로 저장
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

        # PID 제어기 생성 및 설정
        self.linear_pid = PIDController(kp=0.01, ki=0.0, kd=0.1, max_output=self.objConfiguration.getConfiguration('max_speed'))
        self.angular_pid = PIDController(kp=0.01, ki=0.0, kd=0.1, max_output=self.objConfiguration.getConfiguration('max_yaw_rate'))

    def motion_to_target(self, current_position, target_position, current_yaw, dt):
            """
            현재 위치에서 목표 위치까지 이동을 위한 모션 계산
            """
            dx = target_position[0] - current_position[0]
            dy = target_position[1] - current_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            target_yaw = math.atan2(dy, dx)

            angular_velocity = (target_yaw - current_yaw) / dt
            linear_velocity = distance / dt

            # 각도 및 위치 업데이트
            new_yaw = current_yaw + angular_velocity * dt
            new_x = current_position[0] + linear_velocity * math.cos(new_yaw) * dt
            new_y = current_position[1] + linear_velocity * math.sin(new_yaw) * dt



            return [new_x, new_y, new_yaw]

    
    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "StopSimulation":
            self.setStateValue("mode","DEAD")
            self.continueTimeAdvance()

            return True
        if strPort == "RequestManeuver":
            if self.getStateValue('mode')=="POSITION" and objEvent.dblPositionN == -999999 and objEvent.dblPositionE == -999999 and objEvent.dblYaw == -999999:
                print('STOP')
                # self.setStateValue('mode',"ARRIVE")
                self.isNavi = False

            else:
                self.isNavi = True
                self.setStateValue('mode',"POSITION")
                self.setStateValue("target_x", objEvent.dblPositionN)
                self.setStateValue("target_y", objEvent.dblPositionE)
                self.setStateValue("target_yaw", objEvent.dblYaw)

        return True

    def funcOutput(self):
        if self.getStateValue('mode') == 'POSITION':
            objRequestMessage = MsgCurrentPose(self.ID, 
                                                self.getStateValue('current_position_x'), 
                                                self.getStateValue('current_position_y'),
                                                self.getStateValue('current_position_yaw'),
                                                self.getStateValue('current_position_lin_vel'),
                                                self.getStateValue('current_position_ang_vel'),
                                                self.getTime()
                                                )
            self.addOutputEvent("ManeuverState", objRequestMessage)
        return True 
    
    # def funcInternalTransition(self):
    #     if self.getStateValue('mode') == 'POSITION':
    #         # X = self.motion([self.getStateValue('current_position_x'),
    #         #                 self.getStateValue('current_position_y'),
    #         #                 self.getStateValue('current_position_yaw'),
    #         #                 self.getStateValue('current_position_lin_vel'),
    #         #                 self.getStateValue('current_position_ang_vel')], 
    #         #                 float(self.getStateValue('target_linearVelocity')), 
    #         #                 float(self.getStateValue('target_angularVelocity')),
    #         #                 float(self.getStateValue('dt')))
    #         # self.setStateValue('current_position_x', X[0])
    #         # self.setStateValue('current_position_y', X[1])
    #         # self.setStateValue('current_position_yaw', X[2])
    #         # self.setStateValue('current_position_lin_vel', X[3])
    #         # self.setStateValue('current_position_ang_vel', X[4])
    #         current_position = [self.getStateValue('current_position_x'), self.getStateValue('current_position_y')]
    #         target_position = [self.getStateValue('target_x'), self.getStateValue('target_y')]
    #         current_yaw = self.getStateValue('target_yaw')

    #         new_pos = self.motion_to_target(current_position, target_position, current_yaw, self.dt)
    #         self.setStateValue('current_position_x', new_pos[0])
    #         self.setStateValue('current_position_y', new_pos[1])
    #         self.setStateValue('current_position_yaw', new_pos[2])
    #     return True
    def funcInternalTransition(self):
        if self.getStateValue('mode') == 'POSITION' and self.isNavi == True:
            current_position = [self.getStateValue('current_position_x'), self.getStateValue('current_position_y')]
            target_position = [self.getStateValue('target_x'), self.getStateValue('target_y')]
            current_yaw = self.getStateValue('target_yaw')

            distance_error = math.sqrt((target_position[0] - current_position[0])**2 + (target_position[1] - current_position[1])**2)
            angle_to_target = math.atan2(target_position[1] - current_position[1], target_position[0] - current_position[0])
            angle_error = (angle_to_target - current_yaw + math.pi) % (2 * math.pi) - math.pi

            linear_velocity = self.linear_pid.control(distance_error, self.dt)
            angular_velocity = self.angular_pid.control(angle_error, self.dt)

            new_yaw = current_yaw + angular_velocity * self.dt
            new_x = current_position[0] + linear_velocity * math.cos(new_yaw) * self.dt
            new_y = current_position[1] + linear_velocity * math.sin(new_yaw) * self.dt

            self.setStateValue('current_position_x', new_x)
            self.setStateValue('current_position_y', new_y)
            self.setStateValue('current_position_yaw', new_yaw)
            self.setStateValue('current_position_lin_vel', linear_velocity)
            self.setStateValue('current_position_ang_vel', angular_velocity)
        else:
            pass
    

    def funcTimeAdvance(self):
        if self.getStateValue('mode') == 'POSITION':
            return 0.1
        else:
            return 9999999
        

    def funcSelect(self):
        pass
    pass
