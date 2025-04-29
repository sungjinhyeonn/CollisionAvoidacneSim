import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from Models.Atomic.Local_Planner import LPP
from SimulationEngine.Utility.Configurator import Configurator

class APFPlanner:
    def __init__(self):
        # APF 파라미터 추가
        self.k_att = 1.0      # 인력 계수
        self.k_rep = 500.0    # 척력 계수
        self.rho_0 = 8.0      # 장애물 영향 범위
        
        # Configurator 설정
        self.config = Configurator()
        self.config.addConfiguration('robot_radius', 3)
        self.config.addConfiguration('max_speed', 1.0)
        self.config.addConfiguration('min_speed', 0.0)
        self.config.addConfiguration('max_yaw_rate', 10 * math.pi / 180.0)
        self.config.addConfiguration('dt', 0.2)
        self.config.addConfiguration('predict_time', 3)
        self.config.addConfiguration('safety_tolerance', 5)
        
        # Local_Planner 인스턴스 생성
        self.local_planner = LPP("TestRobot", self.config)
        
        # 초기 상태
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_yaw = math.pi/4
        self.goal = np.array([25.0, 25.0])
        
        # 장애물 설정
        self.obstacles = {
            'obs1': np.array([6.0, 6.0]),
            'obs2': np.array([15.0, 15.0]),
            'obs3': np.array([24.0, 6.0]),
            'obs4': np.array([6.0, 24.0])
        }
        
        # Local_Planner 초기 상태 설정
        self.local_planner.curPose = [self.robot_pos[0], self.robot_pos[1], self.robot_yaw, 0, 0]
        self.local_planner.goal = (self.goal[0], self.goal[1])
        self.local_planner.obstacles = {k: (v[0], v[1]) for k, v in self.obstacles.items()}
        
        # Matplotlib 설정
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()
        
    def setup_plot(self):
        self.ax.set_xlim(-5, 30)
        self.ax.set_ylim(-5, 30)
        self.ax.grid(True)
        
    def step(self):
        # Local_Planner의 calc_apf_action 호출
        target_state = self.local_planner.calc_apf_action()
        
        if target_state is None:
            return False
            
        # 상태 업데이트
        self.robot_pos[0] = target_state[0]
        self.robot_pos[1] = target_state[1]
        self.robot_yaw = target_state[2]
        
        # Local_Planner의 상태도 업데이트
        self.local_planner.curPose = [self.robot_pos[0], self.robot_pos[1], self.robot_yaw, 0, 0]
        
        # 목표 도달 체크
        dist_to_goal = np.linalg.norm(self.goal - self.robot_pos)
        return dist_to_goal < 1.0
    
    def visualize(self):
        self.ax.clear()
        self.setup_plot()
        
        # 그리드 생성
        x = np.linspace(-5, 30, 35)
        y = np.linspace(-5, 30, 35)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        # 벡터장 계산
        for i in range(len(x)):
            for j in range(len(y)):
                pos = np.array([X[i,j], Y[i,j]])
                att, rep = self.calc_forces(pos)
                total = att + 3.0 * rep
                
                # 벡터 정규화
                magnitude = np.linalg.norm(total)
                if magnitude > 0:
                    total = total / magnitude
                
                U[i,j] = total[0]
                V[i,j] = total[1]
        
        # 벡터장 그리기
        self.ax.quiver(X, Y, U, V, alpha=0.3)
        
        # 로봇 그리기
        robot_arrow_length = 1.0
        dx = robot_arrow_length * math.cos(self.robot_yaw)
        dy = robot_arrow_length * math.sin(self.robot_yaw)
        self.ax.arrow(self.robot_pos[0], self.robot_pos[1], dx, dy,
                     head_width=0.5, head_length=0.8, fc='blue', ec='blue')
        
        # 목표점 그리기
        self.ax.plot(self.goal[0], self.goal[1], 'g*', markersize=15, label='Goal')
        
        # 장애물 그리기
        for obs_pos in self.obstacles.values():
            circle = plt.Circle((obs_pos[0], obs_pos[1]), 1.0, color='red', alpha=0.5)
            self.ax.add_artist(circle)
        
        # 현재 힘 벡터 표시
        att, rep = self.calc_forces(self.robot_pos)
        total = att + 3.0 * rep
        if np.linalg.norm(total) > 0:
            total = total / np.linalg.norm(total)
            self.ax.quiver(self.robot_pos[0], self.robot_pos[1],
                         total[0], total[1],
                         color='red', scale=10, width=0.005)
        
        plt.draw()
        plt.pause(0.01)
    
    def calc_forces(self, pos):
        # 인력 계산 (거리에 비례하도록 수정)
        to_goal = self.goal - pos
        dist_to_goal = np.linalg.norm(to_goal)
        att_force = self.k_att * to_goal / dist_to_goal  # 단위 벡터로 정규화
        
        # 척력 계산 (지수적으로 증가하도록 수정)
        rep_force = np.zeros(2)
        min_obstacle_dist = float('inf')
        
        for obs_pos in self.obstacles.values():
            to_robot = pos - obs_pos
            dist = np.linalg.norm(to_robot)
            min_obstacle_dist = min(min_obstacle_dist, dist)
            
            if dist <= self.rho_0:
                n = to_robot / dist  # 장애물로부터의 단위 벡터
                # 지수적으로 증가하는 척력
                rep_magnitude = self.k_rep * np.exp(-dist/2) * (1.0/dist - 1.0/self.rho_0)
                rep_force += rep_magnitude * n
        
        # 장애물이 매우 가까울 때 척력을 더 강화
        if min_obstacle_dist < self.rho_0/2:
            rep_force *= (self.rho_0/min_obstacle_dist)**2
        
        # 전체 힘 (척력에 더 높은 가중치)
        total_force = att_force + 5.0 * rep_force
        
        return att_force, rep_force

def main():
    planner = APFPlanner()
    
    while True:
        if planner.step():
            print("Goal reached!")
            break
        planner.visualize()
        
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main() 