from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgManeuverState import MsgManeuverState
from Models.Message.MsgGoal import MsgGoal
from Models.Message.MsgDone import MsgDone
from shapely.geometry import Point, Polygon, LineString
import numpy as np
from Models.Atomic.GPP import RRT  # RRT 클래스를 임포트
from Models.Atomic.Neural_RRT import Neural_RRT


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as pltPolygon

class GPP(DEVSAtomicModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)
        self.objConfiguration = objConfiguration
        self.terrain_polygons = objConfiguration.getConfiguration('terrain_polygons')
        self.starts = objConfiguration.getConfiguration('agent_starts')
        self.start = (self.starts[0][0], self.starts[0][1])  # 초기 위치
        self.goal = None
        self.path = []

        # 상태 변수
        self.addStateVariable("mode", "GPP")  # 초기 상태는 WAIT
        self.addStateVariable("replan", True)


        # Ports
        self.addInputPort("GlobalDestination")
        self.addInputPort("Replan")
        self.addInputPort("OtherManeuverState")
        self.addInputPort("MyManeuverState")
        self.addInputPort("StopSimulation_IN")
        self.addOutputPort("GlobalWaypoint")  # 경로 전달용 포트
        self.addOutputPort("Done")  # 도착 완료 신호

    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "Goal_IN" and self.getStateValue("mode") == "GPP":
            self.goal = (objEvent.goal_x, objEvent.goal_y)
            self.setStateValue("mode", "GPP")  # 상태를 경로 생성 모드로 변경

        elif strPort == "StopSimulation_IN":
            # 시뮬레이션 종료 시 초기화
            self.setStateValue("mode", "WAIT")

        elif strPort == "Replan":

            self.setStateValue("replan", True)
            self.start = [objEvent.dblPositionE, objEvent.dblPositionN]
        
        elif strPort == "OtherManeuverState" or strPort == "MyManeuverState":
            # 재계획 또는 상태 업데이트 이벤트가 발생한 경우
            self.update_data(objEvent)
        self.continueTimeAdvance()

    def funcInternalTransition(self):
        mode = self.getStateValue("mode")
        
        if mode == "GPP" and self.getStateValue("replan") and self.goal is not None:
            # 경로 생성 후 전송 준비
            # self.path = self.generate_path(self.start, self.goal, self.terrain_polygons)
            # ... existing code ...
            self.path = Neural_RRT(self.start, self.goal, self. terrain_polygons)
            self.setStateValue("replan", False)

        # 현재 위치와 최종 목적지 간의 거리 계산
        if self.goal and self.start:
            distance_to_goal = np.linalg.norm(np.array(self.start) - np.array(self.goal))
            target_tolerance = 2
            if distance_to_goal < target_tolerance:
                # 목표에 도달한 경우 stopSim 메시지 전송
                print("Arrived at the goal. Sending stopSim.")
                
                self.setStateValue("mode", "DONE")
                self.goal =None

        return True
    def funcOutput(self):
        # GPP 모드에서만 경로 전송
        if self.getStateValue("mode") == "GPP" and self.path:
            
            x, y = self.path[0]
            msg = MsgManeuverState(self.ID, y, x)
            self.addOutputEvent("GlobalWaypoint", msg)  # 각 좌표를 GlobalWaypoint 포트를 통해 전송
            # self.setStateValue("mode", "WAIT")  # 경로 생성 후 대기 상태로 전환
            self.path =None
        elif self.getStateValue("mode") == "DONE":
            objRequestMessage = MsgDone(self.ID, 
                            self.start[0], 
                            self.start[1])
            self.addOutputEvent("Done", objRequestMessage)
            self.setStateValue("mode", "GPP")

        return True

    def funcTimeAdvance(self):
        mode = self.getStateValue('mode')
        
        if mode == "WAIT":
            # WAIT 상태에서는 무한 대기
            # return float('inf')
            return 99999999
        elif mode == "GPP":
            # GPP 상태에서는 즉시 전이
            return 0.1
        else:
            # 그 외의 상태에서는 기본 대기 시간 설정 (예: 5초)
            return 1

    def generate_path(self, start, goal, terrain_polygons):
        # RRT 객체를 생성하고 경로를 계산
        raw_path = RRT(start, goal, terrain_polygons)

        # RRT 결과를 단순화 (LOS를 사용하여 경로 간소화)
        if raw_path:
            simplified_path = simplify_path_with_los(raw_path, terrain_polygons)
            # 경로가 한 점일 경우 처리
            if len(simplified_path) == 1:
                self.path = simplified_path
            else:
                self.path = simplified_path[1:]  # 시작 지점 제외
        else:
            self.path = []

        # 경로 시각화 (선택 사항)
        # plot_path_with_terrain(start, goal, self.path, terrain_polygons)

        return self.path  # 최종 경로 반환

    
    def update_data(self, objEvent):
        # 관측된 데이터 갱신 로직 추가
        pass

def plot_path_with_terrain(start, goal, path, terrain_polygons):
    fig, ax = plt.subplots()
    
    # Plot terrain polygons
    for polygon in terrain_polygons:
        poly_patch = pltPolygon(polygon, closed=True, color='green', alpha=0.5)
        ax.add_patch(poly_patch)
    
    # Plot start and goal
    ax.plot(start[0], start[1], 'go', label='Start')
    ax.plot(goal[0], goal[1], 'ro', label='Goal')
    
    # Plot path
    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')
    
    # Set axis limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    plt.show()

def simplify_path_with_los(path, terrain_polygons):
    simplified_path = [path[0]]
    current_index = 0

    while current_index < len(path) - 1:
        last_visible = current_index + 1
        for next_index in range(current_index + 2, len(path)):
            if line_of_sight(path[current_index], path[next_index], terrain_polygons):
                last_visible = next_index
            else:
                break
        simplified_path.append(path[last_visible])
        current_index = last_visible

    if simplified_path[-1] != path[-1]:
        simplified_path.append(path[-1])
    
    return simplified_path

def line_of_sight(p1, p2, terrain_polygons):
    line = LineString([p1, p2])
    for polygon in terrain_polygons:
        if line.intersects(Polygon(polygon)):
            return False
    return True
