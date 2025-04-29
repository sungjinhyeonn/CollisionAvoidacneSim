from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgGoal import MsgRequestManeuver
from Models.Message.MsgControl import MsgRequestManeuverControl
from Models.Message.MsgStopSimulation import MsgStopSimulation
import math
import re
import pulp as lp
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
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, -1.0))  # 작업 완수 최대화, 분배 균형 최대화, 이동 시간 최소화
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
        self.agent_assignments = {}
        self.initialize_goals()
        # self.visualize_assignments()

    
    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "MyManeuverState":
            pass

        elif strPort == "OtherManeuverState" and self.getStateValue("mode") != 'ARRIVE':
            # self.setStateValue('mode', 'ASSIGN')
            print(objEvent)
            pass

        elif strPort == "DoneReport" and self.getStateValue("mode") == 'SEND':
            agent_id = objEvent.strID
            # self.assign_goal(agent_id, objEvent)  # strID에 해당하는 에이전트에 새 목표 할당
            try:
                self.assign_next_goal(agent_id)
            except:
                print('goal empty')
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

    '''
    선형 프로그래밍
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

    #     # KMeans 클러스터링 실행
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
    # def initialize_goals(self):
    #     # KMeans 클러스터링을 사용하여 각 에이전트에게 할당할 목적지 결정
    #     kmeans = KMeans(n_clusters=self.numAgent, random_state=0)
    #     kmeans.fit(self.agent_goals)
        
    #     # 각 클러스터에 속하는 목적지를 에이전트에 할당
    #     for i in range(self.numAgent):
    #         indices = np.where(kmeans.labels_ == i)[0]
    #         if len(indices) > 1:
    #             tasks = np.array(self.agent_goals)[indices]
    #             ordered_tasks = self.solve_tsp(tasks)
    #         else:
    #             ordered_tasks = np.array(self.agent_goals)[indices]

    #         agent_id = f'Agent_{i}_Planner'
    #         self.agents_goals[agent_id] = ordered_tasks.tolist()

    #     # 초기 목적지 할당
    #     for agent_id in self.agents_goals:
    #         self.assign_next_goal(agent_id)

    # def assign_next_goal(self, agent_id):
    #     # 다음 목적지 할당
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
    #     # TSP를 해결하는 로직
    #     if len(tasks) < 2:
    #         return tasks
    #     dist_matrix = cdist(tasks, tasks)
    #     row_ind, col_ind = linear_sum_assignment(dist_matrix)
    #     return tasks[col_ind]
                
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

    def initialize_goals(self):
        population = self.toolbox.population(n=300)
        NGEN = 40
        
        for gen in range(NGEN):
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.5, mutpb=0.2)
            fits = self.toolbox.map(self.toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = self.toolbox.select(offspring, k=len(population))

        best_ind = tools.selBest(population, 1)[0]
        for i in range(self.numAgent):
            tasks = np.array(self.agent_goals)[np.where(np.array(best_ind) == i)[0]]
            if len(tasks) > 1:
                ordered_tasks = self.solve_tsp(tasks, self.agents[i][:2])
            else:
                ordered_tasks = tasks
            self.agents_goals[f'Agent_{i}_Planner'] = list(ordered_tasks)
            self.assign_next_goal(f'Agent_{i}_Planner')

    def assign_next_goal(self, agent_id):
        if self.agents_goals[agent_id]:
            next_goal = self.agents_goals[agent_id].pop(0)
            goal_x, goal_y = next_goal
            if (goal_x, goal_y) not in self.visited_goals:
                objRequestMessage = MsgRequestManeuver(agent_id, goal_x, goal_y)
                self.pending_output_events.append(('Goal_OUT', objRequestMessage))
                self.visited_goals.add((goal_x, goal_y))
            else:
                print(f"Goal already visited.")
        else:
            print(f"No more goals left for {agent_id}")

    def solve_tsp(self, tasks, agent_start):
        # 에이전트 시작 위치를 작업 목록에 추가
        all_points = np.vstack([agent_start, tasks])
        
        # TSP를 위한 거리 행렬 계산
        dist_matrix = squareform(pdist(all_points))

        # 헝가리안 알고리즘을 사용하여 최적의 할당을 찾음
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # 최적 경로 반환
        ordered_tasks = all_points[col_ind]

        # 결과 시각화
        plt.figure(figsize=(10, 8))
        plt.scatter(all_points[:, 0], all_points[:, 1], color='red')  # 작업 위치 표시
        plt.scatter(all_points[col_ind[0], 0], all_points[col_ind[0], 1], color='blue', s=100, label='Start')  # 시작 위치 강조

        # 경로를 화살표로 연결
        for i in range(1, len(col_ind)):
            start_pos = all_points[col_ind[i - 1]]
            end_pos = all_points[col_ind[i]]
            plt.arrow(start_pos[0], start_pos[1], end_pos[0] - start_pos[0], end_pos[1] - start_pos[1], 
                    head_width=2, length_includes_head=True, color='k')

        plt.title('Traveling Salesman Problem with Agent Start - Best Path Found')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()

        return ordered_tasks[1:]  # 에이전트 시작 위치를 제외한 목적지 반환
    
    def visualize_assignments(self):
        plt.figure(figsize=(10, 8))
        colors = ['r', 'g', 'b', 'y', 'm']
        for i, agent in enumerate(self.agents):
            if i in self.agent_assignments:
                tasks_assigned = self.agent_assignments[i]
                plt.scatter(agent[0], agent[1], color=colors[i % len(colors)], s=100, label=f'Agent {i+1}')
                plt.scatter(tasks_assigned[:, 0], tasks_assigned[:, 1], color=colors[i % len(colors)], s=50)
                for task in tasks_assigned:
                    plt.plot([agent[0], task[0]], [agent[1], task[1]], 'k--', linewidth=0.5)

        plt.title('Task Allocation and Route Visualization')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()