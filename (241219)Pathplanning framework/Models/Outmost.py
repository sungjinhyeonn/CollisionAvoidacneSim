from SimulationEngine.ClassicDEVS.DEVSCoupledModel import DEVSCoupledModel
from Models.Agent.Agent import Agent
from Models.Obstacles.Obstacle import Obstacle
from Models.SimController.SimContoroller import SimController
from Models.Message.MsgDone import MsgDone
from Models.GCS.GCS import GCS

class Outmost(DEVSCoupledModel):
    def __init__(self, objConfiguration):
        super().__init__("SimModel")
        self.objConfiguration = objConfiguration
        # Retrieve configuration parameters
        self.scenario_num= objConfiguration.getConfiguration('scenario_num')
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
        self.numAgents = len(objConfiguration.getConfiguration('agent_starts'))
        self.numObstacles = objConfiguration.getConfiguration('numObstacles')

        agent_goals = objConfiguration.getConfiguration('agent_goals')
        # Create multiple agents
        self.agents = []
        agent_starts = objConfiguration.getConfiguration('agent_starts')
        agent_goals = objConfiguration.getConfiguration('agent_goals')
        
        for i in range(self.numAgents):
            agent = Agent(f'Agent_{i}', objConfiguration)
            self.agents.append(agent)
            self.addModel(agent)

        # Create GCS
        self.gcs = GCS('GCS_1', objConfiguration)
        self.addModel(self.gcs)
        
        # Create simulation controller
        self.controller = SimController('SimController', objConfiguration)
        self.addModel(self.controller)

        # Create multiple obstacles
        self.obstacles = []
        for i in range(self.numObstacles):
            obstacle = Obstacle(f'Obstacle_{i}', objConfiguration)
            self.obstacles.append(obstacle)
            self.addModel(obstacle)

        # Add couplings
        for agent in self.agents:
            self.addCoupling(self.gcs, "SetGoal", agent, "Goal_IN")
            self.addCoupling(agent, "Done_OUT", self.gcs, "DoneReport")
            self.addCoupling(agent, "Done_OUT", self.controller, "DoneReport")
            self.addCoupling(agent, "ManeuverState_OUT", self.controller, "ManeuverState_IN")
            self.addCoupling(agent, "ManeuverState_OUT", self.gcs, "ManeuverState_IN")
            self.addCoupling(self.controller, "StopSimulation_OUT", agent, f"{agent.ID}_StopSimulation")

            for obstacle in self.obstacles:
                self.addCoupling(obstacle, "ManeuverState_OUT", agent, "ManeuverState_IN")
                self.addCoupling(obstacle, "ManeuverState_OUT", self.controller, "ManeuverState_IN")

        # Add couplings between agents
        for i, agent in enumerate(self.agents):
            for j, other_agent in enumerate(self.agents):
                if i != j:
                    self.addCoupling(other_agent, "ManeuverState_OUT", agent, "ManeuverState_IN")
        
        # GCS와의 StopSimulation 연결
        self.addCoupling(self.controller, "StopSimulation_OUT", self.gcs, "StopSimulation_IN")

        # 장애물에 대한 StopSimulation 연결을 마지막에 한 번만 추가
        for obstacle in self.obstacles:
            self.addCoupling(self.controller, "StopSimulation_OUT", obstacle, f'{obstacle.ID}_StopSimulation_IN')
    pass
