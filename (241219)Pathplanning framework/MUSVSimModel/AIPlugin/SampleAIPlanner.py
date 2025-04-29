from MUSVSimModel.AIPlugin.AIPlanner import AIPlanner
import math

class SampleAIPlanner(AIPlanner):

    # do whatever initialization you need to do
    def __init__(self):
        self.intIdxWaypoint = 0

    # return a list of
    # [
    #  targetMovePositionN,       # Move Position N
    #  targetMovePositionE,       # Move Position E
    #  targetFireID,              # Fire Target string ID
    #  targetFirePositionN,       # Fire Position N
    #  targetFirePositionE,       # Fire Position E
    #  done,                      # done == True : end of the behavior
    #  ]
    # If there is no need to move or fire, put None in the element
    # This method will be called in a passive manner by the MUSV simulator
    # objGrid : grid map of the region
    # dicState : dictionary of a single usv's states
    def produceBehavior(self,dicState,objGrid):
        if dicState["numBullet"] == 0 and len(dicState["otherUSVID"]) != 0:
            strMode = 'NAVIGATION'
        elif dicState["numBullet"] > 0 and len(dicState["otherUSVID"]) != 0:
            strMode = 'INTERCEPT'
        else:
            strMode = 'NAVIGATION'

        myPositionN = dicState["myPositionN"]
        myPositionE = dicState["myPositionE"]

        otherPositionN = dicState["otherPositionN"]
        otherPositionE = dicState["otherPositionE"]

        blnDone = False
        if strMode == "NAVIGATION":
            lstWaypoint = dicState["Waypoint"]

            dblTargetN = lstWaypoint[self.intIdxWaypoint][0]
            dblTargetE = lstWaypoint[self.intIdxWaypoint][1]

            if math.sqrt(pow(myPositionN - dblTargetN, 2) + \
                         pow(myPositionE - dblTargetE, 2)) < dicState["WaypointPoximity"]:
                self.intIdxWaypoint = self.intIdxWaypoint + 1
                if self.intIdxWaypoint >= len(lstWaypoint):
                    blnDone = True
                    self.intIdxWaypoint = len(lstWaypoint) - 1
                else:
                    dblTargetN = lstWaypoint[self.intIdxWaypoint][0]
                    dblTargetE = lstWaypoint[self.intIdxWaypoint][1]

        strFireTargetID = None
        dblFireTargetN = None
        dblFireTargetE = None
        if strMode == "INTERCEPT":
            if otherPositionN != None:
                dblTargetN = otherPositionN[0]
                dblTargetE = otherPositionE[0]

            lstEnemies = dicState["otherUSVID"]
            for i in range(len(lstEnemies)):
                deltaN = otherPositionN[i] - myPositionN
                deltaE = otherPositionE[i] - myPositionE
                dblDistance = math.sqrt(math.pow(deltaN,2) + math.pow(deltaE,2))

                if dicState["numBullet"] > 0:
                    if dblDistance <= dicState['dblEffectRange']:
                        strFireTargetID = lstEnemies[i]
                        dblFireTargetN = otherPositionN[i]
                        dblFireTargetE = otherPositionE[i]

        return [\
            dblTargetN, \
            dblTargetE, \
            strFireTargetID,        \
            dblFireTargetN, \
            dblFireTargetE, \
            blnDone,                \
            ]
