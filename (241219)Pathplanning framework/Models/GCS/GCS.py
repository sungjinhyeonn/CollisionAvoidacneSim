from SimulationEngine.ClassicDEVS.DEVSCoupledModel import DEVSCoupledModel
from Models.Atomic.Sensor import Sensor
from Models.Atomic.Planner_GCS import Planner_GCS

class GCS(DEVSCoupledModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)
        self.objConfiguration = objConfiguration
        
        # Initialize Sensor and Planner models within GCS
        self.sensor = Sensor(ID + '_Sensor', objConfiguration)
        self.planner = Planner_GCS(ID + '_Planner', objConfiguration)
        
        # Add models to GCS
        self.addModel(self.sensor)
        self.addModel(self.planner)
        
        # Define input and output ports based on given configuration
        
        self.addInputPort("ManeuverState_IN")  # Receive ManeuverState from agents
        self.addOutputPort("SetGoal")  # Send goal settings to agents
        self.addOutputPort("StopSimulation_OUT")  # To handle stop simulation signals

        # Internal couplings
        self.addCoupling(self, "ManeuverState_IN", self.sensor, "ManeuverState_IN")
        self.addCoupling(self.sensor, "ManeuverState_OUT", self.planner, "OtherManeuverState")
        self.addCoupling(self.planner, "Goal_OUT", self, "SetGoal")
        self.addCoupling(self, "DoneReport", self.planner, "DoneReport")

        # For stopping simulation
        self.addInputPort("StopSimulation_IN")
        self.addCoupling(self, "StopSimulation_IN", self.planner, "StopSimulation_IN")
        self.addCoupling(self, "StopSimulation_IN", self.sensor, "StopSimulation_IN")
        self.addCoupling(self.planner, "StopSimulation_OUT", self, "StopSimulation_OUT")
