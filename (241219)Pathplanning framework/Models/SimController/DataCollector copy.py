from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgCurPose import MsgCurrentPose
import math
import numpy as np
from Models.Message.MsgStopSimulation import MsgStopSimulation
import csv
import re
import os

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

    def add_pose(self, pose):
        # 각 pose에 timestamp 추가
        if pose.strID not in self.data:
            self.data[pose.strID] = []
        self.data[pose.strID].append((pose.x, pose.y, pose.yaw, pose.lin_vel, pose.ang_vel, pose.time))

    def save_to_csv(self, directory):
        for strID, poses in self.data.items():
            filename = f"{directory}/{strID}.csv"
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
        self.agent_id = 'Agent_Maneuver'  # ID of the agent to track

        self.addInputPort("ManeuverState_IN")
        self.addInputPort("DoneReport")
        self.addInputPort("StopSimulation")
        self.addInputPort("MyManeuverState")
        self.addOutputPort("ManeuverState_OUT")
        
        self.addOutputPort(self.ID + "_StopSimulation")  # Add output port for StopSimulation
        self.addStateVariable("mode", "ACTIVE")
        self.pose_storage = PoseStorage()
        self.target_tolerance = 0.5
        self.starts = objConfiguration.getConfiguration('agent_starts')
        self.goals = objConfiguration.getConfiguration('agent_goals')
        self.goal_count = len(self.goals)
        self.doneCount = 0
        self.numAgent = self.objConfiguration.getConfiguration('numAgent')
        # Initialize a dictionary to keep track of agents' states
        self.agent_states = {}

    def has_reached_destination(self, pose):
        if "Agent" in pose.strID:  # Only check the position of agents
            agent_index = int(extract_numbers(pose.strID))
            goal = self.goals[agent_index]
            distance = math.sqrt((pose.x - goal[0]) ** 2 + (pose.y - goal[1]) ** 2)
            print(f'{pose.strID}: distance: {distance}')
            
            return distance < self.target_tolerance # Threshold for reaching the destination
        return False

    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "StopSimulation":
            self.setStateValue("mode", "DEAD")
            self.continueTimeAdvance()
        elif strPort == 'ManeuverState_IN':
            self.pose_storage.add_pose(objEvent)
            if objEvent.strID[0:5] == 'Agent':
                self.agent_states[objEvent.strID] = "ACTIVE"
            # if self.has_reached_destination(objEvent):
            #     self.agent_states[objEvent.strID] = "STOP"
            #     if all(state == "STOP" for state in self.agent_states.values()):
            #         self.setStateValue("mode", "STOP")
            #         self.pose_storage.save_to_csv("log")
        elif strPort == "DoneReport":
            # Increment the DoneReport counter for the reporting agent
            self.goals.pop(0)
            if len(self.goals) == 0:
                self.setStateValue("mode", "STOP")
                self.pose_storage.save_to_csv("log")


            self.continueTimeAdvance()
        elif strPort == "MyManeuverState":
            print('MyManeuverState')
        
        return True
    
    def funcOutput(self):
        if self.getStateValue("mode") == "STOP":
            objStopSimulation = MsgStopSimulation(self.ID)
            self.addOutputEvent(self.ID + "_StopSimulation", objStopSimulation)
        return True
    
    def funcInternalTransition(self):
        # 터미널 클리어하기
        clear_terminal()
        return True
    
    def funcTimeAdvance(self):
        if self.getStateValue('mode') == "ACTIVE":
            return 0.1
        else:
            return 999999999
    
    def funcSelect(self):
        pass
