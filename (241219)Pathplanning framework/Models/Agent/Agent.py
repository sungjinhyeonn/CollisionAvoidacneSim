from SimulationEngine.ClassicDEVS.DEVSCoupledModel import DEVSCoupledModel
from Models.Atomic.Maneuver import Maneuver
from Models.Atomic.Sensor import Sensor
from Models.Atomic.Global_Planner import GPP
from Models.Atomic.Local_Planner import LPP
from Models.Atomic.Planner_GCS import *  # GCS 모델을 포함합니다.

class Agent(DEVSCoupledModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)

        # 입력 및 출력 포트 설정
        self.addInputPort("ManeuverState_IN")
        self.addInputPort(ID + "_StopSimulation")
        self.addInputPort("Goal_IN")

        self.addOutputPort("ManeuverState_OUT")
        self.addOutputPort("Done_OUT")

        # 하위 모델 생성
        self.objSensor = Sensor(ID + '_Sensor', objConfiguration)
        self.objGPP = GPP(ID + '_GPP', objConfiguration)
        self.objLPP = LPP(ID + '_LPP', objConfiguration)
        self.objManeuver = Maneuver(ID + '_Maneuver', objConfiguration)
        # self.objGCS = Planner_GCS(ID + '_GCS', objConfiguration)  # GCS 추가

        # 하위 모델 추가
        self.addModel(self.objSensor)
        self.addModel(self.objGPP)
        self.addModel(self.objLPP)
        self.addModel(self.objManeuver)
        # self.addModel(self.objGCS)

        # Couplings 설정
        # GPP와 LPP 간 Coupling
        self.addCoupling(self, "Goal_IN", self.objGPP, "Goal_IN")
        self.addCoupling(self.objGPP, "GlobalWaypoint", self.objLPP, "GlobalWaypoint")
        self.addCoupling(self.objLPP, "Replan", self.objGPP, "Replan")

        # LPP와 Maneuver 간 Coupling
        self.addCoupling(self.objLPP, "RequestManeuver", self.objManeuver, "RequestManeuver")

        # 센서와 Planner 간 Coupling
        self.addCoupling(self.objManeuver, "ManeuverState", self, "ManeuverState_OUT")
        self.addCoupling(self.objManeuver, "ManeuverState", self.objSensor, "MyManeuverState")
        self.addCoupling(self.objManeuver, "ManeuverState", self.objLPP, "MyManeuverState")


        self.addCoupling(self.objSensor, "ManeuverState_OUT", self.objLPP, "OtherManeuverState")


        
        self.addCoupling(self, "ManeuverState_IN",self.objSensor, "ManeuverState_IN" )

        # 종료 및 도착 신호 Coupling
        self.addCoupling(self.objGPP, "Done", self, "Done_OUT")  # LPP의 Done 메시지를 GCS로 전달
        self.addCoupling(self, ID + "_StopSimulation", self.objGPP, "StopSimulation_IN")
        self.addCoupling(self, ID + "_StopSimulation", self.objLPP, "StopSimulation_IN")
        self.addCoupling(self, ID + "_StopSimulation", self.objManeuver, "StopSimulation_IN")
        self.addCoupling(self, ID + "_StopSimulation", self.objSensor, "StopSimulation_IN")

        # Done 메시지 전달을 위한 coupling
        self.addCoupling(self.objGPP, "Done_OUT", self, "Done_OUT")  # GPP의 Done 메시지를 Agent의 Done_OUT으로 전달

