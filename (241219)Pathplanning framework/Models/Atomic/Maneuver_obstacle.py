import re
import math
import numpy as np
from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgCurPose import MsgCurrentPose

class Maneuver_obstacle(DEVSAtomicModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)
        self.objConfiguration = objConfiguration

        # 입력 및 출력 포트 추가
        self.addInputPort("RequestManeuver")
        self.addInputPort("StopSimulation_IN")
        self.addOutputPort("ManeuverState")

        # 설정 값 로드
        self.dt = objConfiguration.getConfiguration('dt')
        self.addStateVariable('dt', self.dt)
        self.addStateVariable("mode", "POSITION")

        # 장애물 위치, 목표, 방향 설정
        obstacle_positions = objConfiguration.getConfiguration('obstacle_positions')
        obstacle_targets = objConfiguration.getConfiguration('obstacle_targets')
        obstacle_yaws = objConfiguration.getConfiguration('obstacle_yaws')
        self.obstacle_speed = objConfiguration.getConfiguration('obstacle_speed')

        # ID에서 숫자 추출하여 인덱스 설정
        obstacle_index = int(re.search(r'\d+', ID).group())
        self.targets = obstacle_targets[obstacle_index]
        self.current_target_index = 0

        # 초기 상태 변수 설정
        self.addStateVariable('current_position_x', obstacle_positions[obstacle_index][0])
        self.addStateVariable('current_position_y', obstacle_positions[obstacle_index][1])
        self.addStateVariable('current_position_yaw', obstacle_yaws[obstacle_index])
        self.addStateVariable('current_position_lin_vel', 0)
        self.addStateVariable('current_position_ang_vel', 0)

        # 첫 번째 목표 위치 설정
        self.update_target()

        # 물리적 제약 파라미터 추가
        self.max_acceleration = 0.5  # 최대 가속도
        self.max_deceleration = 0.8  # 최대 감속도
        self.current_velocity = 0.0
        self.prev_velocity = 0.0

    def update_target(self):
        current_target = self.targets[self.current_target_index]
        self.setStateValue("target_x", current_target[0])
        self.setStateValue("target_y", current_target[1])

    def motion(self, x, lin_vel, ang_vel, dt):
        """물리적 제약을 고려한 운동 모델 업데이트"""
        # 가속도 제한
        velocity_diff = lin_vel - self.prev_velocity
        if velocity_diff > 0:
            max_diff = self.max_acceleration * dt
            if velocity_diff > max_diff:
                lin_vel = self.prev_velocity + max_diff
        else:
            max_diff = self.max_deceleration * dt
            if abs(velocity_diff) > max_diff:
                lin_vel = self.prev_velocity - max_diff

        # 위치 업데이트
        new_x = x[0] + lin_vel * math.cos(x[2]) * dt
        new_y = x[1] + lin_vel * math.sin(x[2]) * dt

        x[0] = new_x
        x[1] = new_y
        x[2] += ang_vel * dt
        x[3] = lin_vel
        x[4] = ang_vel

        self.prev_velocity = lin_vel
        return x

    def calculate_control(self, current_x, current_y, current_yaw, target_x, target_y):
        """제약을 고려한 제어 입력 계산"""
        desired_yaw = math.atan2(target_y - current_y, target_x - current_x)
        yaw_diff = desired_yaw - current_yaw
        
        # 각도 정규화
        if yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        elif yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi
        
        # 부드러운 회전을 위한 각속도 계산
        ang_vel = 0.05 * yaw_diff
        
        # 거리 기반 속도 계산
        distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
        
        # 목표 지점 근처에서 감속
        if distance < 2.0:
            target_speed = self.obstacle_speed * (distance / 2.0)
        else:
            target_speed = self.obstacle_speed

        return target_speed, ang_vel

    def funcExternalTransition(self, strPort, objEvent):
        """ 외부 전이 함수 """
        if strPort == "StopSimulation_IN":
            print(f"{self.ID}: Received StopSimulation message")
            self.setStateValue("mode", "WAIT")
            self.continueTimeAdvance()
        elif strPort == "RequestManeuver":
            self.setStateValue("mode", "POSITION")

    def funcOutput(self):
        """ 출력 함수 """
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

    def funcInternalTransition(self):
        """ 내부 전이 함수 """
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
            
            if math.sqrt((X[0] - target_x) ** 2 + (X[1] - target_y) ** 2) < 0.5:
                if self.current_target_index < len(self.targets) - 1:
                    self.current_target_index += 1
                    self.update_target()
                else:
                    self.current_target_index = 0  # 처음으로 돌아가기
                    self.update_target()
        # elif self.getStateValue('mode') == 'ARRIVE':
        #     pass
    def funcTimeAdvance(self):
        """ 시간 진행 함수 """
        if self.getStateValue('mode') == "WAIT":
            return float('inf')
        elif self.getStateValue('mode') == 'POSITION':
            return 0.2
        else:
            return float('inf')

    def funcSelect(self):
        pass
