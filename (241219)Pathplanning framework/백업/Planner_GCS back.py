from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgGoal import MsgRequestManeuver
from Models.Message.MsgControl import MsgRequestManeuverControl
from Models.Message.MsgStopSimulation import MsgStopSimulation
import math
import re
from enum import Enum
from heapq import heappop, heappush
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as pltPolygon
from shapely.geometry import Point, LineString, Polygon

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


class Planner_GCS(DEVSAtomicModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)
        self.objLogger = None
        self.ID = ID
        #ros 스레드
        self.objConfiguration = objConfiguration

        self.addInputPort("OtherManeuverState")
        self.addInputPort("StopSimulation_IN")
        self.addInputPort('DoneReport')
        self.addOutputPort('Goal_OUT')

        # 설정 값을 가져와서 속성으로 저장
        self.robot_radius = objConfiguration.getConfiguration('robot_radius')
        self.max_speed = objConfiguration.getConfiguration('max_speed')
        self.min_speed = objConfiguration.getConfiguration('min_speed')
        self.max_yaw_rate = objConfiguration.getConfiguration('max_yaw_rate')
        self.max_accel = objConfiguration.getConfiguration('max_accel')
        self.max_delta_yaw_rate = objConfiguration.getConfiguration('max_delta_yaw_rate')
        self.v_resolution = objConfiguration.getConfiguration('v_resolution')
        self.yaw_rate_resolution = objConfiguration.getConfiguration('yaw_rate_resolution')
        self.dt = objConfiguration.getConfiguration('dt')
        self.predict_time = objConfiguration.getConfiguration('predict_time')
        self.to_goal_cost_gain = objConfiguration.getConfiguration('to_goal_cost_gain')
        self.speed_cost_gain = objConfiguration.getConfiguration('speed_cost_gain')
        self.obstacle_cost_gain = objConfiguration.getConfiguration('obstacle_cost_gain')
        self.heuristic_cost_gain = objConfiguration.getConfiguration('heuristic_cost_gain')
        self.robot_type = objConfiguration.getConfiguration('robot_type')
        self.robot_stuck_flag_cons = objConfiguration.getConfiguration('robot_stuck_flag_cons')
        self.terrain_polygons= objConfiguration.getConfiguration('terrain_polygons')
        self.agent_goals = self.objConfiguration.getConfiguration('agent_goals')
        self.numAgent = self.objConfiguration.getConfiguration('numAgent')
        
        self.addStateVariable("mode", 'ASSIGN')

        # 출력 이벤트를 임시 저장할 리스트
        self.pending_output_events = []
        self.visited_goals = set()

        # 초기에 각 에이전트에게 목표를 할당
        self.initialize_goals()

    
    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "MyManeuverState":
            pass

        elif strPort == "OtherManeuverState" and self.getStateValue("mode") != 'ARRIVE':
            # self.setStateValue('mode', 'ASSIGN')
            print(objEvent)
            pass

        elif strPort == "DoneReport" and self.getStateValue("mode") == 'SEND':
            agent_id = objEvent.strID
            self.assignNewGoal(agent_id, objEvent)  # strID에 해당하는 에이전트에 새 목표 할당

            self.setStateValue('mode', 'ASSIGN')
            
        return True

    def funcOutput(self):
        if self.getStateValue('mode') == 'ASSIGN':

            # 각 에이전트에게 해당 목표 위치를 할당
            if self.getStateValue('mode') == 'ASSIGN':
                while self.pending_output_events:
                    port, message = self.pending_output_events.pop(0)
                    self.addOutputEvent(port, message)#topic
            # 모드를 'SEND'로 변경하여 모든 목표가 할당
            self.setStateValue('mode', 'SEND')
        return True

    def funcInternalTransition(self):
       
        return True
    
    def funcTimeAdvance(self):
        if self.getStateValue('mode') == "ASSIGN" or self.getStateValue('mode') == "ARRIVE":
            return 0.1
        else:
            return 999999999
    
    def funcSelect(self):
        pass

    def assignNewGoal(self, agent_id, objEvent):
        current_position = [objEvent.goal_x, objEvent.goal_y]
        
        # 목적지 리스트를 튜플로 변환하여 `visited_goals`에서 체크
        goals_positions = [tuple(goal) for goal in self.agent_goals if tuple(goal) not in self.visited_goals]
        
        if not goals_positions:
            print("No unvisited goals available")
            return

        dist_matrix = distance_matrix([current_position], goals_positions)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        if col_ind.size > 0:
            nearest_goal_index = col_ind[0]
            goal = goals_positions[nearest_goal_index]
            self.agent_goals.remove(list(goal))  # 목적지 리스트에서 해당 목적지 삭제 (목적지를 리스트로 다시 변환)
            self.assign_goal(agent_id, list(goal))  # 목적지를 리스트로 다시 변환하여 할당
        else:
            print("No feasible path to unvisited goals")
            
    def initialize_goals(self):
        for i in range(self.numAgent):
            # if i < len(self.agent_goals):/
            goal = self.agent_goals.pop(0)
            agent_id = f'Agent_{i}_Planner'
            self.assign_goal(agent_id, goal)

    def assign_goal(self, agent_id, goal):
        goal_x, goal_y = goal

        if (goal_x, goal_y) not in self.visited_goals:
            objRequestMessage = MsgRequestManeuver(agent_id, goal_x, goal_y)
            self.pending_output_events.append(('Goal_OUT', objRequestMessage))
            self.visited_goals.add((goal_x, goal_y))
        else:
            print(f"Goal {goal} already visited.")

'''
    def assignNewGoal(self, agent_id, objEvent=None):
        # agent_id에 대응되는 새 목표를 설정
        index = int(agent_id.split('_')[1])
        # if 0 <= index < len(self.agent_goals):
        if  len(self.agent_goals) != 0:
            goal = self.agent_goals.pop(0)
            goal_x, goal_y = goal
            objRequestMessage = MsgRequestManeuver(agent_id, goal_x, goal_y)
            self.pending_output_events.append(('Goal_OUT', objRequestMessage))
    
    def initialize_goals(self):
        for i in range(self.numAgent):
            # 각 에이전트에게 목표 할당
            if i < len(self.agent_goals):  # 에이전트 수보다 목적지가 적을 수 있으므로 검사
                goal = self.agent_goals.pop(0)
                agent_id = f'Agent_{i}_Planner'
                goal_x, goal_y = goal
                objRequestMessage = MsgRequestManeuver(agent_id, goal_x, goal_y)
                self.pending_output_events.append(('Goal_OUT', objRequestMessage))
            else:
                # 목적지가 부족한 경우 처리
                print(f"No available goal for Agent_{i}_Planner")

    '''