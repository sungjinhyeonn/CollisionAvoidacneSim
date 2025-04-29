from SimulationEngine.ClassicDEVS.DEVSCoupledModel import DEVSCoupledModel
from Models.Atomic.Maneuver_obstacle import *
from Models.Atomic.Sensor import *
# from Models.Atomic.Planner import *

class Obstacle(DEVSCoupledModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)

        self.addInputPort(f"{ID}_StopSimulation_IN")

        self.addOutputPort("ManeuverState_OUT")


        self.objManeuver = Maneuver_obstacle(ID, objConfiguration)
        self.addModel(self.objManeuver)


        self.addCoupling(self.objManeuver, "ManeuverState", self, "ManeuverState_OUT")
        self.addCoupling(self, f"{ID}_StopSimulation_IN", self.objManeuver, "StopSimulation_IN")
        print(f"Obstacle {ID}: Added StopSimulation coupling")