from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgCurPose import MsgCurrentPose
import math
import numpy as np
import re

def extract_numbers(input_string):
    # 정규 표현식을 사용하여 문자열에서 숫자만 추출
    numbers = re.findall(r'\d+', input_string)
    # 추출된 숫자를 하나의 문자열로 연결하여 반환
    return ''.join(numbers)

class Maneuver(DEVSAtomicModel):
    def __init__(self, ID, objConfiguration):

        super().__init__(ID)
        self.objConfiguration = objConfiguration

        self.addInputPort("RequestManeuver")
        self.addInputPort("StopSimulation")

        self.addOutputPort("ManeuverState")

        self.id = int(extract_numbers(self.ID))
        starts = objConfiguration.getConfiguration('agent_starts')


        # 설정 값을 가져와서 속성으로 저장
        self.dt = objConfiguration.getConfiguration('dt')
        self.addStateVariable('dt', self.dt)
        self.addStateVariable("mode", "WAIT")
        self.addStateVariable('current_postion_x', starts[self.id][0])
        self.addStateVariable('current_postion_y', starts[self.id][1])
        self.addStateVariable('current_postion_yaw', starts[self.id][2])
        self.addStateVariable('current_postion_lin_vel', starts[self.id][3])
        self.addStateVariable('current_postion_ang_vel', starts[self.id][4])

        self.curPose = starts[self.id]  # 초기값을 필요에 따라 설정

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
    
    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "StopSimulation":
            self.setStateValue("mode","DEAD")
            self.continueTimeAdvance()

            return True
        if strPort == "RequestManeuver":
            self.setStateValue("mode","POSITION")
            self.setStateValue("target_linearVelocity", objEvent.lin_vel)
            self.setStateValue("target_angularVelocity", objEvent.ang_vel)

        return True

    def funcOutput(self):
        if self.getStateValue('mode') == 'POSITION':
            objRequestMessage = MsgCurrentPose(self.ID, 
                                                self.getStateValue('current_postion_x'), 
                                                self.getStateValue('current_postion_y'),
                                                self.getStateValue('current_postion_yaw'),
                                                self.getStateValue('current_postion_lin_vel'),
                                                self.getStateValue('current_postion_ang_vel')
                                                )
            self.addOutputEvent("ManeuverState", objRequestMessage)
        return True 
    
    def funcInternalTransition(self):
        if self.getStateValue('mode') == 'POSITION':
            X = self.motion([self.getStateValue('current_postion_x'),
                            self.getStateValue('current_postion_y'),
                            self.getStateValue('current_postion_yaw'),
                            self.getStateValue('current_postion_lin_vel'),
                            self.getStateValue('current_postion_ang_vel')], 
                            float(self.getStateValue('target_linearVelocity')), 
                            float(self.getStateValue('target_angularVelocity')),
                            float(self.getStateValue('dt')))
            self.setStateValue('current_postion_x', X[0])
            self.setStateValue('current_postion_y', X[1])
            self.setStateValue('current_postion_yaw', X[2])
            self.setStateValue('current_postion_lin_vel', X[3])
            self.setStateValue('current_postion_ang_vel', X[4])
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
