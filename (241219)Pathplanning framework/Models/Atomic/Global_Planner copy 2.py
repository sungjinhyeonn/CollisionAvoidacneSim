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
        self.agent_id = ID.split('_')[1]  # Agent_1_GPP에서 1을 추출
        self.start = (self.starts[int(self.agent_id)][0], self.starts[int(self.agent_id)][1])
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
        self.addOutputPort("Done_OUT")  # Done 메시지를 GCS로 전달하기 위한 포트

        # Neural RRT 인스턴스 생성 (한 번만 생성)
        self.neural_rrt = Neural_RRT()
        self.successful_paths = []  # 성공한 경로 저장
        self.training_interval = 100  # 100번의 경로 생성마다 학습
        self.path_count = 0

    def funcExternalTransition(self, strPort, objEvent):
        message_agent_id = objEvent.strID.split('_')[1]

        if strPort == "Goal_IN" and self.getStateValue("mode") == "GPP":
            if message_agent_id == self.agent_id:  # 에이전트 번호만 비교
                self.goal = (objEvent.goal_x, objEvent.goal_y)
                self.setStateValue("mode", "GPP")

        elif strPort == "StopSimulation_IN":
            self.setStateValue("mode", "WAIT")

        elif strPort == "Replan":
            if message_agent_id == self.agent_id:  # 에이전트 번호만 비교
                self.setStateValue("replan", True)
                self.start = [objEvent.dblPositionE, objEvent.dblPositionN]
        
        elif strPort == "OtherManeuverState":
            # 다른 에이전트의 상태 정보는 무시
            pass
        elif strPort == "MyManeuverState":
            if message_agent_id == self.agent_id:  # 에이전트 번호만 비교
                self.update_data(objEvent)
            
        self.continueTimeAdvance()

    def funcInternalTransition(self):
        mode = self.getStateValue("mode")
        
        if mode == "GPP" and self.getStateValue("replan") and self.goal is not None:
            self.path = self.generate_path(self.start, self.goal, self.terrain_polygons)
            
            if not self.path:  # 경로를 찾지 못한 경우
                print("No path found, using direct path")
                self.path = [(self.goal[0], self.goal[1])]
            
            self.setStateValue("replan", False)

        # 현재 위치와 최종 목적지 간의 거리 계산
        if self.goal and self.start:
            distance_to_goal = np.linalg.norm(np.array(self.start) - np.array(self.goal))
            target_tolerance = 2
            if distance_to_goal < target_tolerance:
                # 목표에 도달한 경우 Done 메시지 전송
                print(f"Agent {self.agent_id} arrived at goal: {self.goal}")
                self.setStateValue("mode", "DONE")
                self.goal = None

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
            # Done 메시지를 GCS로 전달
            objRequestMessage = MsgDone(self.ID, 
                            self.start[0], 
                            self.start[1])
            self.addOutputEvent("Done_OUT", objRequestMessage)  # Done_OUT 포트로 메시지 전송
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
        try:
            # terrain_polygons를 Shapely Polygon 객체로 변환
            polygons = [Polygon(p) for p in terrain_polygons]
            
            # 안전 마진 추가 (선반과의 거리를 좀 더 확보)
            safety_margin = 3.0
            expanded_polygons = []
            for polygon in polygons:
                expanded_polygons.append(polygon.buffer(safety_margin))
            
            # 시작점과 목표점이 장애물 내부에 있는지 확인
            start_point = Point(start)
            goal_point = Point(goal)
            
            for polygon in expanded_polygons:
                if polygon.contains(start_point) or polygon.contains(goal_point):
                    print("Start or goal point is inside obstacle!")
                    return []
            
            # 확장된 다각형을 사용하여 RRT 경로 계획
            raw_path = RRT(start, goal, expanded_polygons)
            
            if raw_path:
                # Line of Sight를 이용한 경로 단순화
                simplified_path = simplify_path_with_los(raw_path, expanded_polygons)
                if len(simplified_path) == 1:
                    self.path = simplified_path
                else:
                    self.path = simplified_path[1:]
                
                # 경로 시각화 (디버깅용)
                plot_path_with_terrain(start, goal, simplified_path, terrain_polygons)
                
                return self.path
            else:
                print("No path found between start and goal")
                return []
            
        except Exception as e:
            print(f"Error in generate_path: {e}")
            return []

    
    def update_data(self, objEvent):
        # 관측된 데이터 갱신 로직 추가
        pass

    def train_neural_rrt(self):
        """성공한 경로들을 사용하여 Neural RRT 학습"""
        print(f"Training Neural RRT with {len(self.successful_paths)} paths")
        
        total_samples = 0
        for path_data in self.successful_paths:
            total_samples += len(path_data['path']) - 1
            
        print(f"Total training samples: {total_samples}")
        
        for path_data in self.successful_paths:
            start = path_data['start']
            goal = path_data['goal']
            path = path_data['path']
            
            # 경로의 각 점에 대해 학습 데이터 생성
            for i in range(len(path)-1):
                current = path[i]
                next_point = path[i+1]
                
                # 데이터 수집
                self.neural_rrt.collect_data(
                    np.array(start),
                    np.array(goal),
                    np.array(next_point),
                    np.array(current)
                )
        
        # 학습 수행
        self.neural_rrt.train(epochs=5, batch_size=32)
        print(f"Current training data size: {len(self.neural_rrt.training_data)}")
        print("Training completed")
        
        # 메모리 관리: 오래된 경로 제거
        max_paths = 1000
        if len(self.successful_paths) > max_paths:
            self.successful_paths = self.successful_paths[-max_paths:]

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
