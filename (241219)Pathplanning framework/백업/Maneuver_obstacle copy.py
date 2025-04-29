import re
import math
import numpy as np
from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgCurPose import MsgCurrentPose

class Maneuver_obstacle(DEVSAtomicModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)
        self.objConfiguration = objConfiguration

        self.addInputPort("RequestManeuver")
        self.addInputPort("StopSimulation")

        self.addOutputPort("ManeuverState")

        # 설정 값을 가져와서 속성으로 저장
        self.dt = objConfiguration.getConfiguration('dt')
        self.addStateVariable('dt', self.dt)
        self.addStateVariable("mode", "POSITION")

        # 현재 위치, 헤딩, 목표 위치를 설정 파일에서 가져오기
        self.obstacle_positions = objConfiguration.getConfiguration('obstacle_positions')
        self.obstacle_targets = objConfiguration.getConfiguration('obstacle_targets')
        self.obstacle_yaws = objConfiguration.getConfiguration('obstacle_yaws')

        # ID가 문자열로 전달되는 경우 숫자 부분만 추출
        obstacle_index = int(re.search(r'\d+', ID).group())

        # 현재 목표지점 인덱스
        self.current_target_index = 0

        # 초기 위치 및 목표 위치 설정
        self.addStateVariable('current_position_x', self.obstacle_positions[obstacle_index][0])
        self.addStateVariable('current_position_y', self.obstacle_positions[obstacle_index][1])
        self.addStateVariable('current_position_yaw', self.obstacle_yaws[obstacle_index])
        self.addStateVariable('current_position_lin_vel', 0)
        self.addStateVariable('current_position_ang_vel', 0)
        self.addStateVariable("target_x", self.obstacle_targets[self.current_target_index][0])
        self.addStateVariable("target_y", self.obstacle_targets[self.current_target_index][1])


        self.curPose = [
            float(self.getStateValue('current_position_x')),
            float(self.getStateValue('current_position_y')),
            float(self.getStateValue('current_position_yaw')),
            float(self.getStateValue('current_position_lin_vel')),
            float(self.getStateValue('current_position_ang_vel'))
        ]  # 초기값을 필요에 따라 설정

    def motion(self, x, lin_vel, ang_vel, dt):
        """
        motion model
        """
        x[2] += ang_vel * dt
        x[0] += lin_vel * math.cos(x[2]) * dt
        x[1] += lin_vel * math.sin(x[2]) * dt
        x[3] = lin_vel
        x[4] = ang_vel

        return x

    def calculate_control(self, current_x, current_y, current_yaw, target_x, target_y):
        """
        선속도와 각속도를 계산하는 함수
        """
        # 목표 위치까지의 각도
        desired_yaw = math.atan2(target_y - current_y, target_x - current_x)
        # 현재 각도와 목표 각도 사이의 차이
        yaw_diff = desired_yaw - current_yaw

        # 각속도 제어
        if yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        elif yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi

        # 각속도는 목표 각도로 회전하도록 설정
        ang_vel = 0.1 * yaw_diff

        # 목표 위치까지의 거리
        distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)

        # 선속도는 목표 위치로 이동하도록 설정
        lin_vel = 0.5 if distance > 0.5 else 0

        return lin_vel, ang_vel

    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "StopSimulation":
            self.setStateValue("mode", "DEAD")
            self.continueTimeAdvance()
            return True
        if strPort == "RequestManeuver":
            self.setStateValue("mode", "POSITION")
        return True

    def funcOutput(self):
        if self.getStateValue('mode') == 'POSITION':
            objRequestMessage = MsgCurrentPose(self.ID,
                                               self.getStateValue('current_position_x'),
                                               self.getStateValue('current_position_y'),
                                               self.getStateValue('current_position_yaw'),
                                               self.getStateValue('current_position_lin_vel'),
                                               self.getStateValue('current_position_ang_vel')
                                               )
            self.addOutputEvent("ManeuverState", objRequestMessage)
        return True

    def funcInternalTransition(self):
        if self.getStateValue('mode') == 'POSITION':
            current_x = self.getStateValue('current_position_x')
            current_y = self.getStateValue('current_position_y')
            current_yaw = self.getStateValue('current_position_yaw')
            target_x = self.getStateValue("target_x")
            target_y = self.getStateValue("target_y")

            lin_vel, ang_vel = self.calculate_control(current_x, current_y, current_yaw, target_x, target_y)
            self.curPose = [current_x, current_y, current_yaw, 0, 0] 
            X = self.motion(self.curPose, lin_vel, ang_vel, self.dt)

            self.setStateValue('current_position_x', X[0])
            self.setStateValue('current_position_y', X[1])
            self.setStateValue('current_position_yaw', X[2])
            self.setStateValue('current_position_lin_vel', X[3])
            self.setStateValue('current_position_ang_vel', X[4])

            # 목표지점에 도달했는지 확인
            if math.sqrt((X[0] - target_x) ** 2 + (X[1] - target_y) ** 2) < 0.5:
                if self.current_target_index < len(self.obstacle_targets) - 1:
                    self.current_target_index += 1
                    new_target_x = self.obstacle_targets[self.current_target_index][0]
                    new_target_y = self.obstacle_targets[self.current_target_index][1]
                    self.setStateValue("target_x", new_target_x)
                    self.setStateValue("target_y", new_target_y)
                    print(f"Moving to next target: ({new_target_x}, {new_target_y})")
        return True


    def funcTimeAdvance(self):
        if self.getStateValue('mode') == 'POSITION':
            return 0.1
        else:
            return 9999999999999
        pass

    def funcSelect(self):
        pass
    pass
