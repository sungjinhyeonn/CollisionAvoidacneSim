from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgGoal import MsgGoal
from Models.Message.MsgControl import MsgRequestManeuverControl
from Models.Message.MsgStopSimulation import MsgStopSimulation
import math
import re
# import pulp as lp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from enum import Enum
from heapq import heappop, heappush
import numpy as np
import matplotlib.pyplot as plt
from deap import creator, base, tools, algorithms
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import re

def extract_numbers_from_id(input_string):
    """ 문자열에서 숫자만 추출 """
    return int(''.join(re.findall(r'\d+', input_string)))


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
        self.agents = self.objConfiguration.getConfiguration('agent_starts')
        
        self.addStateVariable("mode", 'ASSIGN')
        # DEAP setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, -10.0))  # 작업 완수 최대화, 분배 균형 최대화, 이동 시간 최소화
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", np.random.randint, 0, self.numAgent, len(self.agent_goals))
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.fitness_func)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.numAgent-1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        self.pending_output_events = []
        self.visited_goals = set()
        self.agents_goals = {}
        self.next_goals = {}
        self.task_queues = {}
        self.initialize_goals()
        # self.visualize_assignments()

    
    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "StopSimulation_IN":
            print(f"{self.ID}: Received StopSimulation message")
            self.setStateValue("mode", "WAIT")  # DEAD -> WAIT
            self.continueTimeAdvance()
        elif strPort == 'ManeuverState_IN':
            # self.pose_storage.add_pose(objEvent)
            # if objEvent.strID[0:5] == 'Agent':
            #     self.agent_states[objEvent.strID] = "ACTIVE"
            pass
        elif strPort == "DoneReport":
            agent_id = objEvent.strID
            print(f"Received DoneReport from {agent_id}")
            
            if agent_id in self.agents_goals:
                current_task = self.agents_goals[agent_id]
                
                if len(current_task) == 2:  # 피킹 완료
                    # 적재 위치로 이동
                    next_goal = current_task[1]
                    self.agents_goals[agent_id] = [next_goal]  # 적재 위치만 남김
                    objRequestMessage = MsgGoal(agent_id, next_goal[0], next_goal[1])
                    self.pending_output_events.append(('Goal_OUT', objRequestMessage))
                    print(f"{agent_id} completed picking, moving to placing: {next_goal}")
                    
                elif len(current_task) == 1:  # 적재 완료
                    print(f"{agent_id} completed placing")
                    
                    # task_queues에서 다음 피킹-적재 작업 가져오기
                    if self.task_queues[agent_id]:
                        next_task = self.task_queues[agent_id].pop(0)  # 다음 피킹-적재 쌍 가져오기
                        self.agents_goals[agent_id] = next_task  # 새 작업 전체 할당
                        next_goal = next_task[0]  # 새 피킹 포인트
                        objRequestMessage = MsgGoal(agent_id, next_goal[0], next_goal[1])
                        self.pending_output_events.append(('Goal_OUT', objRequestMessage))
                        print(f"{agent_id} starting new task: Picking at {next_goal} -> Placing at {next_task[1]}")
                        print(f"Remaining tasks in queue: {len(self.task_queues[agent_id])}")
                    else:
                        self.agents_goals[agent_id] = []
                        print(f"No more tasks in queue for {agent_id}")
                
                self.setStateValue("mode", "ASSIGN")

    def funcOutput(self):
        if self.getStateValue('mode') == 'ASSIGN':

            # 각 에이전트에게 해당 목표 위치를 할당
            if self.getStateValue('mode') == 'ASSIGN':
                while self.pending_output_events:
                    port, message = self.pending_output_events.pop(0)
                    self.addOutputEvent(port, message)#topic
            # 모드를 'SEND'로 변경���여 모든 목표가 할당
            self.setStateValue('mode', 'SEND')
        return True

    def funcInternalTransition(self):
       
        return True
    
    def funcTimeAdvance(self):
        if self.getStateValue('mode') == "WAIT":
            return float('inf')
        elif self.getStateValue('mode') == "ASSIGN" or self.getStateValue('mode') == "ARRIVE":
            return 0.1
        else:
            return 999999999
    
    def funcSelect(self):
        pass

    '''
    형 프로그래밍
    '''
    # def initialize_goals(self):
    #     # self.agents와 self.agent_goals를 np.array로 변환
    #     if not isinstance(self.agents, np.ndarray):
    #         self.agents = np.array(self.agents, dtype=float).reshape(-1, 2)
    #     if not isinstance(self.agent_goals, np.ndarray):
    #         self.agent_goals = np.array(self.agent_goals, dtype=float)

    #     # 배열이 올바른 형태인지 검사
    #     if self.agents.ndim != 2 or self.agent_goals.ndim != 2:
    #         raise ValueError("Agents and agent_goals must be 2-dimensional arrays.")

    #     # 두 배열의 열 수가 같은지 확인
    #     if self.agents.shape[1] != self.agent_goals.shape[1]:
    #         raise ValueError("Agents and goals must have the same number of columns.")

    #     # KMeans 클러스터 실행
    #     kmeans = KMeans(n_clusters=self.numAgent, random_state=0)
    #     kmeans.fit(self.agent_goals)

    #     # 각 클러스터에 속하는 목적지를 에이전트에 할당
    #     for i in range(self.numAgent):
    #         indices = np.where(kmeans.labels_ == i)[0]
    #         if len(indices) > 1:
    #             tasks = self.agent_goals[indices]
    #             ordered_tasks = self.solve_tsp(tasks)
    #         else:
    #             ordered_tasks = self.agent_goals[indices]

    #         agent_id = f'Agent_{i}_Planner'
    #         self.agents_goals[agent_id] = list(ordered_tasks)

    #     # 초기 목적지 할당
    #     for agent_id in self.agents_goals:
    #         self.assign_next_goal(agent_id)

    # def assign_next_goal(self, agent_id):
    #     # 다음 목적지 할당 로직
    #     if self.agents_goals[agent_id]:
    #         next_goal = self.agents_goals[agent_id].pop(0)
    #         goal_x, goal_y = next_goal
    #         if (goal_x, goal_y) not in self.visited_goals:
    #             objRequestMessage = MsgRequestManeuver(agent_id, goal_x, goal_y)
    #             self.pending_output_events.append(('Goal_OUT', objRequestMessage))
    #             self.visited_goals.add((goal_x, goal_y))
    #         else:
    #             print(f"Goal {goal_x, goal_y} already visited.")
    #     else:
    #         print(f"No more goals left for {agent_id}")

    # def solve_tsp(self, tasks):
    #     # TSP 해결 로직
    #     if len(tasks) < 2:
    #         return tasks
    #     dist_matrix = cdist(tasks, tasks)
    #     row_ind, col_ind = linear_sum_assignment(dist_matrix)
    #     return tasks[col_ind]

    '''
    타겟 클러스터링
    '''
    def initialize_goals(self):
        num_tasks = len(self.agent_goals)
        num_agents = self.numAgent
        
        print("\nInitializing task assignments...")
        print(f"Number of agents: {num_agents}")
        print(f"Number of tasks: {num_tasks}")
        
        # 각 에이전트의 시작 위치
        agent_positions = [np.array(agent[:2]) for agent in self.agents]
        
        # 작업 할당을 위한 비용 행렬 생성
        cost_matrix = np.zeros((num_agents, num_tasks))
        for i in range(num_agents):
            for j in range(num_tasks):
                picking_point = np.array(self.agent_goals[j][0])
                cost_matrix[i][j] = np.linalg.norm(agent_positions[i] - picking_point)
        
        # 헝가리안 알고리즘을 사용하여 최적 할당
        agent_indices, task_indices = linear_sum_assignment(cost_matrix)
        
        # 각 에이전트별 작업 큐 초기화
        self.agents_goals = {}
        self.task_queues = {}
        for i in range(num_agents):
            agent_id = f'Agent_{i}_GPP'
            self.task_queues[agent_id] = []
        
        # 작업을 에이전트별로 분배하고 마지막에 출발지 추가
        for i in range(num_agents):
            agent_id = f'Agent_{i}_GPP'
            start_pos = self.agents[i][:2]  # 에이전트의 시작 위치
            agent_tasks = []

            # 이 에이전트에 할당된 작업들 수집
            for task_idx in range(num_tasks):
                if task_idx % num_agents == i:  # 이 에이전트에 할당된 작업
                    task_pair = self.agent_goals[task_idx]
                    agent_tasks.append(task_pair)
                    print(f"\n{agent_id} queued Task {task_idx}:")
                    print(f"  Picking at {task_pair[0]} -> Placing at {task_pair[1]}")

            # 마지막에 시작 위치로 돌아가는 작업 추가 (피킹-적재 쌍 형태로)
            start_pos = [start_pos[0], start_pos[1]]
            agent_tasks.append([start_pos, start_pos])  # 시작 위치를 피킹과 적재 위치로 동일하게 설정
            print(f"\n{agent_id} final return task added:")
            print(f"  Return to start position: ({start_pos[0]}, {start_pos[1]})")
            
            # 작업 큐에 저장
            self.task_queues[agent_id] = agent_tasks
        
        # 각 에이전트의 첫 작업 할당
        for agent_id in self.task_queues:
            if self.task_queues[agent_id]:
                first_task = self.task_queues[agent_id].pop(0)
                self.agents_goals[agent_id] = first_task
                first_goal = first_task[0]
                objRequestMessage = MsgGoal(agent_id, first_goal[0], first_goal[1])
                self.pending_output_events.append(('Goal_OUT', objRequestMessage))
                print(f"\n{agent_id} starting with:")
                print(f"  Initial goal (Picking): ({first_goal[0]}, {first_goal[1]})")
                print(f"  Remaining tasks in queue: {len(self.task_queues[agent_id])}")

    def assign_next_goal(self, agent_id):
        if agent_id not in self.agents_goals:
            print(f"Error: {agent_id} not found in agents_goals")
            return

        current_task = self.agents_goals[agent_id]
        
        if len(current_task) == 2:  # [피킹, 적재]
            # 피킹 포인트로 이동
            next_goal = current_task[0]
            self.agents_goals[agent_id] = [current_task[1]]  # 적재 포인트만 남김
            print(f"{agent_id} -> Picking: {next_goal}")
            
        elif len(current_task) == 1:  # [적재]
            # 적재 포인트로 이동
            next_goal = current_task[0]
            print(f"{agent_id} -> Placing: {next_goal}")
            
            # 다음 작업이 있으면 바로 새 작업 전체를 할당
            if self.task_queues[agent_id]:
                next_task = self.task_queues[agent_id].pop(0)
                self.agents_goals[agent_id] = next_task  # 전체 작업(피킹+적재)을 할당
                print(f"{agent_id} next task ready: {next_task}")
            else:
                self.agents_goals[agent_id] = []  # 모든 작업 완료된 경우만 비움
                print(f"{agent_id} all tasks completed")

        # 다음 목적지로 이동 명령 전송
        objRequestMessage = MsgGoal(agent_id, next_goal[0], next_goal[1])
        self.pending_output_events.append(('Goal_OUT', objRequestMessage))

    def solve_tsp(self, tasks):
        # TSP를 해결하는 로직
        if len(tasks) < 2:
            return tasks
        dist_matrix = cdist(tasks, tasks)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        return tasks[col_ind]
                
    '''
    유전 알고리즘
    '''
    def fitness_func(self, individual):
        assignment = np.array(individual)
        task_count = np.zeros(self.numAgent, dtype=int)
        total_travel_time = 0.0

        for i in range(self.numAgent):
            agent_pos = self.agents[i][:2]  # X, Y 좌표만 사용
            task_indices = np.where(assignment == i)[0]
            task_positions = np.array(self.agent_goals)[task_indices]
            if task_indices.size > 0:
                distances = np.linalg.norm(task_positions - agent_pos, axis=1)
                total_travel_time += np.sum(distances)
                task_count[i] = len(task_indices)

        unassigned_penalty = 1000 if 0 in task_count else 0
        balance_score = -np.std(task_count)

        return (sum(task_count) - unassigned_penalty, balance_score, -total_travel_time)

    # def initialize_goals(self):
    #     population = self.toolbox.population(n=500)
    #     NGEN = 1000
        
    #     for gen in range(NGEN):
    #         offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.5, mutpb=0.2)
    #         fits = self.toolbox.map(self.toolbox.evaluate, offspring)
    #         for fit, ind in zip(fits, offspring):
    #             ind.fitness.values = fit
    #         population = self.toolbox.select(offspring, k=len(population))

    #     best_ind = tools.selBest(population, 1)[0]
        
    #     for i in range(self.numAgent):
    #         tasks = np.array(self.agent_goals)[np.where(np.array(best_ind) == i)[0]]
    #         agent_start = self.agents[i][:2]  # 에이전트의 시작 위치

    #         if len(tasks) > 0:
    #             all_points = np.vstack([tasks])
    #             ordered_tasks = self.solve_tsp_with_ga(all_points)
    #             # 시작점에서 가장 가까운 목적지를 첫 번째 방문지로 설정
    #             start_idx = np.argmin(np.linalg.norm(ordered_tasks - agent_start, axis=1))
    #             # 시작점에 가장 가까운 목적지부터 순서 재배열
    #             ordered_tasks = np.roll(ordered_tasks, -start_idx, axis=0)
    #         else:
    #             ordered_tasks = [agent_start]  # 목적지가 없으면 시작 위치만 반환
    #         # self.visualize_tsp_route(all_points, ordered_tasks)
    #         self.agents_goals[f'Agent_{i}_Planner'] = list(ordered_tasks)
    #         self.visualize_tsp_routes(self.agents ,self.agents_goals)
    #         self.assign_next_goal(f'Agent_{i}_Planner')

    # def assign_next_goal(self, agent_id):
    #     if self.agents_goals[agent_id]:
    #         next_goal = self.agents_goals[agent_id].pop(0)
    #         goal_x, goal_y = next_goal
    #         if (goal_x, goal_y) not in self.visited_goals:
    #             objRequestMessage = MsgRequestManeuver(agent_id, goal_x, goal_y)
    #             self.pending_output_events.append(('Goal_OUT', objRequestMessage))
    #             self.visited_goals.add((goal_x, goal_y))
    #         else:
    #             print(f"Goal {goal_x, goal_y} already visited.")
    #     else:
    #         print(f"No more goals left for {agent_id}")

    # def solve_tsp(self, tasks, agent_start=None):
    #     # 에이전트 시작 위치를 작업 목록에 추가
    #     all_points = np.vstack([agent_start, tasks])
        
    #     # TSP를 위한 거리 행렬 계산
    #     dist_matrix = squareform(pdist(all_points))

    #     # 헝가리안 알고리즘을 사용하여 최적의 할당을 찾음
    #     row_ind, col_ind = linear_sum_assignment(dist_matrix)

    #     # 최적 경로 반환
    #     ordered_tasks = all_points[col_ind]

    #     # 결과 시각화
    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(all_points[:, 0], all_points[:, 1], color='red')  # 작업 위치 표시
    #     plt.scatter(all_points[col_ind[0], 0], all_points[col_ind[0], 1], color='blue', s=100, label='Start')  # 시작 위치 강조

    #     # 경로를 화살표로 연결
    #     for i in range(1, len(col_ind)):
    #         start_pos = all_points[col_ind[i - 1]]
    #         end_pos = all_points[col_ind[i]]
    #         plt.arrow(start_pos[0], start_pos[1], end_pos[0] - start_pos[0], end_pos[1] - start_pos[1], 
    #                 head_width=2, length_includes_head=True, color='k')

    #     plt.title('Traveling Salesman Problem with Agent Start - Best Path Found')
    #     plt.xlabel('X Coordinate')
    #     plt.ylabel('Y Coordinate')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    #     return ordered_tasks[1:]  # 에이전트 시작 위치를 제외한 목적지 반환

    def solve_tsp_with_ga(self, all_points):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("indices", np.random.permutation, len(all_points))
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval_tsp(individual):
            # 경로 길이 계산
            path_length = sum(np.linalg.norm(all_points[individual[i-1]] - all_points[individual[i]]) for i in range(len(individual)))
            return (path_length,)

        toolbox.register("evaluate", eval_tsp)
        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=50)
        hof = tools.HallOfFame(1, similar=np.array_equal)

        # 유전 알고리즘 실행
        algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 30, halloffame=hof)

        best_route_index = hof[0]
        best_route = all_points[best_route_index]

        # 결과 시각화
        # self.visualize_tsp_route(all_points, best_route)

        return best_route

    def visualize_tsp_route(self, all_points, best_route):
        plt.figure(figsize=(10, 8))
        plt.scatter(all_points[:, 0], all_points[:, 1], color='red')  # 모든 위치 표시
        plt.plot(best_route[:, 0], best_route[:, 1], 'o-', color='blue')  # 최적 경로 표시

        plt.title('Optimal TSP Route Found by Genetic Algorithm')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()

    def visualize_tsp_routes(self, agents, agents_goals):
        plt.figure(figsize=(10, 8))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 색상 배열
        for i, (agent_id, goals) in enumerate(agents_goals.items()):
            agent_start = agents[i][:2]  # 에이전트의 시작 위치
            tasks = np.array(goals)  # 할당된 목적지
            # 시작점과 경로 포인트를 연결하는 선 그리기
            plt.plot(*zip(*np.vstack([agent_start, tasks])), marker='o', color=colors[i % len(colors)], label=f'Agent {i+1}')
            plt.scatter(*agent_start, color=colors[i % len(colors)], s=100, zorder=5)
            plt.text(agent_start[0], agent_start[1], f'Agent {i+1}', verticalalignment='bottom', horizontalalignment='right')

        plt.title('Visualized Routes for Each Agent')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_assignments(self):
        plt.figure(figsize=(10, 8))
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'purple', 'brown', 'pink']  # 색상 설정
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H']  # 마커 설정

        for i, agent in enumerate(self.agents):
            agent_pos = agent[:2]  # 에이전트 위치
            tasks = self.agents_goals.get(f'Agent_{i}_Planner', [])
            task_positions = np.array(tasks) if tasks else np.array([])

            # 에이전트 위치 시화
            plt.scatter(agent_pos[0], agent_pos[1], color=colors[i % len(colors)], s=100, marker=markers[i % len(markers)], label=f'Agent {i+1} Start')

            # 할당된 목적지 시각화
            if len(task_positions) > 0:
                plt.scatter(task_positions[:, 0], task_positions[:, 1], color=colors[i % len(colors)], s=50, marker=markers[i % len(markers)])

                # 에이전트와 목적지 연결
                for task in task_positions:
                    plt.plot([agent_pos[0], task[0]], [agent_pos[1], task[1]], 'k--', linewidth=1)

        plt.title('Agent Assignments and Routes')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()