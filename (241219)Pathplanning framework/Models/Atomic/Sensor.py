from SimulationEngine.ClassicDEVS.DEVSAtomicModel import DEVSAtomicModel
from Models.Message.MsgManeuverState import MsgManeuverState

class PoseStorage:
    def __init__(self):
        self.data = []

    def add_or_update_pose(self, pose):
        for i, entry in enumerate(self.data):
            if entry[0] == pose.strID:
                # Update existing entry
                self.data[i] = (pose.strID, pose.x, pose.y, pose.yaw, pose.lin_vel, pose.ang_vel)
                return
        # Add new entry if ID does not exist
        self.data.append((pose.strID, pose.x, pose.y, pose.yaw, pose.lin_vel, pose.ang_vel))

    def __str__(self):
        ret = ""
        for entry in self.data:
            ret += f'ID: {entry[0]}, Position: ({entry[1]}, {entry[2]}), Yaw: {entry[3]}, Linear Velocity: {entry[4]}, Angular Velocity: {entry[5]}\n'
        return ret

class Sensor(DEVSAtomicModel):
    def __init__(self, ID, objConfiguration):
        super().__init__(ID)

        self.objConfiguration = objConfiguration
        self.pose_storage = PoseStorage()
        
        self.addInputPort("ManeuverState_IN")
        self.addInputPort("StopSimulation_IN")
        self.addInputPort("MyManeuverState")
        self.addOutputPort("ManeuverState_OUT")

        self.addStateVariable("mode", 'ACTIVE')


    def funcExternalTransition(self, strPort, objEvent):
        if strPort == "StopSimulation_IN":
            print(f"{self.ID}: Received StopSimulation message")
            self.setStateValue("mode", "WAIT")
            self.continueTimeAdvance()
        elif strPort == 'ManeuverState_IN':
            self.pose_storage.add_or_update_pose(objEvent)

        elif strPort == 'MyManeuverState':
            self.pose_storage.add_or_update_pose(objEvent)
        return True
    
    def funcOutput(self):
        if self.getStateValue('mode')=='ACTIVE' and len(self.pose_storage.data) != 0:
            for i in range(len(self.pose_storage.data)):
                objSensorMessage = MsgManeuverState(self.pose_storage.data[i][0],self.pose_storage.data[i][2],self.pose_storage.data[i][1],self.pose_storage.data[i][3],self.pose_storage.data[i][4])
                self.addOutputEvent("ManeuverState_OUT", objSensorMessage)
        return True
    
    def funcInternalTransition(self):
        return True
    
    def funcTimeAdvance(self):
        if self.getStateValue('mode') == "WAIT":
            return float('inf')
        return 0.1
    
    def funcSelect(self):
        pass