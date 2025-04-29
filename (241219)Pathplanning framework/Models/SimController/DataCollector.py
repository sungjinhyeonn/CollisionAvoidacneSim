from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgCurPose import MsgCurrentPose
import math
import numpy as np
from Models.Message.MsgStopSimulation import MsgStopSimulation
import csv
import re
import os
from datetime import datetime

def clear_terminal():
    # Windows
    if os.name == 'nt':
        os.system('cls')
    # Unix 계열(Linux, macOS)
    else:
        os.system('clear')

def extract_numbers(input_string):
    # 정규 표현식을 사용하여 문자열에서 숫자만 추출
    numbers = re.findall(r'\d+', input_string)
    # 추출된 숫자를 하나의 문자열로 연결하여 반환
    return ''.join(numbers)

class PoseStorage:
    def __init__(self):
        self.data = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_pose(self, pose):
        if pose.strID not in self.data:
            self.data[pose.strID] = []
        self.data[pose.strID].append((pose.x, pose.y, pose.yaw, pose.lin_vel, pose.ang_vel, pose.time))

    def save_to_csv(self, directory):
        for strID, poses in self.data.items():
            filename = f"{directory}/{strID}_{self.timestamp}.csv"
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['x', 'y', 'yaw', 'linear_velocity', 'angular_velocity', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for pose in poses:
                    writer.writerow({
                        'x': pose[0],
                        'y': pose[1],
                        'yaw': pose[2],
                        'linear_velocity': pose[3],
                        'angular_velocity': pose[4],
                        'timestamp': pose[5]
                    })

    def __str__(self):
        ret = ""
        for strID, poses in self.data.items():
            ret += f'ID: {strID}\n'
            for pose in poses:
                ret += f'  Position: ({pose[0]}, {pose[1]}), Yaw: {pose[2]}, Linear Velocity: {pose[3]}, Angular Velocity: {pose[4]}\n'
        return ret

class DataCollector(DEVSAtomicModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)

        self.objConfiguration = objConfiguration
        self.agent_id = 'Agent_Maneuver'

        # 알고리즘 설정 가져오기
        self.use_drl = objConfiguration.getConfiguration('use_drl')
        self.use_dwa = objConfiguration.getConfiguration('use_dwa')
        self.use_dtp = objConfiguration.getConfiguration('use_dtp')
        self.use_apf = objConfiguration.getConfiguration('use_apf')
        self.use_sac = objConfiguration.getConfiguration('use_sac')
        
        # 시나리오 및 반복 실험 정보 가져오기
        self.scenario_num = str(objConfiguration.getConfiguration('scenario'))
        self.iteration_num = str(objConfiguration.getConfiguration('iteration'))
        
        self.drl_checkpoint = objConfiguration.getConfiguration('drl_checkpoint')
        self.sac_model_path = objConfiguration.getConfiguration('sac_model_path')
        
        # 알고리즘 이름 설정
        if self.use_drl:
            self.algorithm = "DDPG_DTP" if self.use_dtp else "DDPG"
        elif self.use_dwa:
            self.algorithm = "DWA"
        elif self.use_apf:
            self.algorithm = "APF"
        elif self.use_sac:
            self.algorithm = "SAC_DTP" if self.use_dtp else "SAC"
        else:
            self.algorithm = "DEFAULT"

        # 로그 디렉토리 설정
        self.log_base = "log"
        scenario_dir = os.path.join(self.log_base, f"scenario_{self.scenario_num}")
        iter_dir = os.path.join(scenario_dir, f"iter_{self.iteration_num}")
        self.log_dir = os.path.join(iter_dir, self.algorithm)
        
        # 디렉토리 생성
        os.makedirs(self.log_dir, exist_ok=True)

        # 로그 디렉토리 설정
        self.addInputPort("ManeuverState_IN")
        self.addInputPort("DoneReport")
        self.addInputPort("StopSimulation_IN")
        self.addInputPort("MyManeuverState")
        self.addOutputPort("ManeuverState_OUT")
        self.addOutputPort("StopSimulation_OUT")
        self.addStateVariable("mode", "ACTIVE")
        self.pose_storage = PoseStorage()
        self.target_tolerance = 0.5
        self.starts = objConfiguration.getConfiguration('agent_starts')
        self.goals = []
        for task_pair in objConfiguration.getConfiguration('agent_goals'):
            self.goals.extend(task_pair)  # 작업 쌍을 풀어서 순차적으로 저장
        self.goal_count = len(self.goals)+(len(self.starts)*2)-1
        self.doneCount = 0
        self.numAgent = self.objConfiguration.getConfiguration('numAgent')
        self.agent_states = {}

    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "StopSimulation_IN":
            self.setStateValue("mode", "STOP")
            self.continueTimeAdvance()
        elif strPort == 'ManeuverState_IN':
            self.pose_storage.add_pose(objEvent)
            if objEvent.strID[0:5] == 'Agent':
                self.agent_states[objEvent.strID] = "ACTIVE"
        elif strPort == "DoneReport":
            self.doneCount += 1
            print(f"Task completed: {self.doneCount}/{self.goal_count}")
            
            if self.doneCount >= self.goal_count:
                print("All tasks completed! Stopping simulation...")
                self.pose_storage.save_to_csv(self.log_dir)
                self.setStateValue("mode", "STOP")
            return True
        elif strPort == "MyManeuverState":
            print('MyManeuverState')
        
        return True
    
    def funcOutput(self):
        if self.getStateValue("mode") == "STOP":
            objStopSimulation = MsgStopSimulation(self.ID)
            self.addOutputEvent("StopSimulation_OUT", objStopSimulation)
            self.setStateValue("mode", "WAIT")
            # 시뮬레이션 엔진에 종료 신호 전달
            self.engine.stop_simulation = True
        return True
    
    def funcInternalTransition(self):
        # 터미널 클리어하기
        clear_terminal()
        return True
    
    def funcTimeAdvance(self):
        if self.getStateValue('mode') == "ACTIVE":
            return 0.1
        elif self.getStateValue('mode') == "STOP":
            return 0.0  # STOP 모드일 때는 즉시 출력
        else:
            return float('inf')
    
    def funcSelect(self):
        pass
