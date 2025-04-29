import sys
sys.path.append('../')
import math
import ast
import csv
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation
import numpy as np
import MUSVSimModel.Utility.PlotUtility as PlotUtility
from MUSVSimModel.Environment.MapModel import GridEnvironment

class LogDisplay:

    def __init__(self,strLogFile,strBackgroundImageFile,\
                 strMapDataFile,dblMeterPerPixel,dblInterval=1.0):
        self.strLogFile = strLogFile
        self.dblCurrentTimestep = 0
        self.dblMeterPerPixel = dblMeterPerPixel
        self.dblInterval = dblInterval
        self.objGrid = GridEnvironment(strMapDataFile, dblMeterPerPixel)
        self.intPixelX,self.intPixelY = self.objGrid.getShape()
        self.intPixelX = self.intPixelX + 1
        self.intPixelY = self.intPixelY + 1

        self.lstAxisRange = [0,0,self.intPixelX*dblMeterPerPixel,self.intPixelY*dblMeterPerPixel]
        if strBackgroundImageFile != None:
            self.objImage = plt.imread('./MapData/mission1.png')
            self.blnBackgroundImage = True
        else:
            self.objImage = None
            self.blnBackgroundImage = False

        objFile = open(strLogFile,'r',newline='')
        objReader = csv.reader(objFile)
        self.dblMaxTime = 0
        self.lstLog = []
        for lstLine in objReader:
            lstLine[0] = float(lstLine[0])
            if lstLine[0] > self.dblMaxTime:
                self.dblMaxTime = lstLine[0]
            self.lstLog.append(lstLine)
        objFile.close()

        self.lstLogWaypoint = []
        for lstLogInstance in self.lstLog:
            if lstLogInstance[2] == 'Waypoint':
                self.lstLogWaypoint.append(lstLogInstance)

        self.lstLogInfo = []
        for lstLogInstance in self.lstLog:
            if lstLogInstance[2] == 'Info':
                self.lstLogInfo.append(lstLogInstance)

        self.lstLogDead = []
        for lstLogInstance in self.lstLog:
            if lstLogInstance[2] == 'Dead':
                self.lstLogDead.append(lstLogInstance)

        self.lstLogDone = []
        for lstLogInstance in self.lstLog:
            if lstLogInstance[2] == 'Done':
                self.lstLogDone.append(lstLogInstance)

        self.dicLogInfo = {}
        for lstLogInstance in self.lstLogInfo:
            dicLogInfoForInstance = {}
            for i in range(5, len(lstLogInstance), 2):
                dicLogInfoForInstance[lstLogInstance[i]] = float(lstLogInstance[i + 1])
            dicLogInfoForInstance[lstLogInstance[3]] = lstLogInstance[4]
            self.dicLogInfo[lstLogInstance[1]] = dicLogInfoForInstance

        # self.fig, self.ax = plt.subplots()
        self.fig = plt.figure()
        plt.rcParams.update({'font.size': 8})
        plt.tight_layout()
        plt.show(block=False)
        # ani = matplotlib.animation.FuncAnimation(fig, self.visualize,interval=80, blit=False)
        # plt.show()

    def getUSVShape(self,dblSize,dblPositionN,dblPositionE,dblYaw):
        # dblYaw = dblYaw - 3.14159/2
        #matCoord = np.asarray([[0,0],[-1,-2],[-1,-3],[1,-3],[1,-2],[0,0]])
        matCoord = np.asarray([[0, 0], [-2, -1], [-3, -1], [-3, 1], [-2, 1], [0, 0]])
        matCoord = np.dot(matCoord,dblSize)
        matRotation = np.asarray([[math.cos(dblYaw),-math.sin(dblYaw)],\
                                 [math.sin(dblYaw),math.cos(dblYaw)]])
        matCoord = np.dot(matRotation,np.transpose(matCoord))
        for i in range(matCoord.shape[1]):
            matCoord[0][i] = matCoord[0][i] + dblPositionE
            matCoord[1][i] = matCoord[1][i] + dblPositionN
        #matCoord = np.transpose(matCoord)
        return matCoord

    def visualizeManeuver(self,lstLogLine):
        if len(lstLogLine) == 0:
            return
        strID = lstLogLine[1]
        if strID.lower().startswith("blue"):
            strColor = 'b'
        else:
            strColor = 'r'

        dicInfo = self.dicLogInfo[strID]

        dicManeuver = {}
        for i in range(3,len(lstLogLine),2):
            dicManeuver[lstLogLine[i]] = float(lstLogLine[i+1])
        matShape = self.getUSVShape(\
            100,dicManeuver['PositionN'],dicManeuver['PositionE'],dicManeuver['Yaw'])
        plt.plot(matShape[0],matShape[1],color=strColor)
        plt.text(dicManeuver['PositionE'],dicManeuver['PositionN'],strID)
        PlotUtility.drawCircle(plt,dicManeuver['PositionE'],dicManeuver['PositionN'], \
                   dicInfo['dblViewRadius'],strColor,":")
        PlotUtility.drawCircle(plt,dicManeuver['PositionE'],dicManeuver['PositionN'], \
                   dicInfo['dblConeDist'],strColor,"--")


    def visualizeWaypoint(self):
        for lstLogWayPointInstance in self.lstLogWaypoint:
            strID = lstLogWayPointInstance[1]
            lstWaypoint = ast.literal_eval(lstLogWayPointInstance[3])
            if strID.lower().startswith("blue"):
                strColor = 'slateblue'
            else:
                strColor = 'darkorange'
            lstN = []
            lstE = []
            for i in range(len(lstWaypoint)):
                lstN.append(lstWaypoint[i][0])
                lstE.append(lstWaypoint[i][1])
                plt.text(lstWaypoint[i][1], lstWaypoint[i][0], \
                         strID+"_"+str(i), style='italic',color=strColor)#,\
                         #bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

            lstLine = plt.plot(lstE,lstN,color=strColor)
            for i in range(len(lstLine)):
                PlotUtility.add_arrow(lstLine[i],color=strColor)

    def visualizeFire(self,lstLog):
        strID = lstLog[1]
        if strID.lower().startswith("blue"):
            strColor = 'slateblue'
        else:
            strColor = 'darkorange'
        dicFire = {}
        for i in range(5,len(lstLog),2):
            dicFire[lstLog[i]] = float(lstLog[i+1])
        dblFromY = dicFire['dblMyPositionN']
        dblFromX = dicFire['dblMyPositionE']
        dblToY = dicFire['dblTargetPositionN']
        dblToX = dicFire['dblTargetPositionE']
        lstLine = plt.plot([dblFromX,dblToX], [dblFromY,dblToY], color=strColor)
        for i in range(len(lstLine)):
            PlotUtility.add_arrow(lstLine[i], color=strColor,size=30)
        # plt.arrow(dblFromX,dblFromY,(dblToX-dblFromX),(dblToY-dblFromY),color=strColor,\
        #           linewidth=2,linestyle='-',arrowprops=dict(arrowstyle="->"))

    def visualizeEndState(self,dblCurrentTime):
        for lstLog in self.lstLogDead:
            dblTimestamp = float(lstLog[0])
            if dblTimestamp > dblCurrentTime:
                continue
            strID = lstLog[1]
            if strID.lower().startswith("blue"):
                strColor = 'slateblue'
            else:
                strColor = 'darkorange'
            dicLog = {}
            for i in range(3,len(lstLog),2):
                dicLog[lstLog[i]] = float(lstLog[i+1])
            dblY = dicLog['dblMyPositionN']
            dblX = dicLog['dblMyPositionE']
            dblYaw = dicLog['dblMyYaw']
            matShape = self.getUSVShape(100, dblY, dblX, dblYaw)
            plt.plot(matShape[0],matShape[1],color=strColor)
            plt.text(dblX,dblY,str(strID+"-Dead"),color='r')
        for lstLog in self.lstLogDone:
            dblTimestamp = float(lstLog[0])
            if dblTimestamp > dblCurrentTime:
                continue
            strID = lstLog[1]
            if strID.lower().startswith("blue"):
                strColor = 'slateblue'
            else:
                strColor = 'darkorange'
            dicLog = {}
            for i in range(3,len(lstLog),2):
                dicLog[lstLog[i]] = float(lstLog[i+1])
            dblY = dicLog['dblMyPositionN']
            dblX = dicLog['dblMyPositionE']
            dblYaw = dicLog['dblMyYaw']
            matShape = self.getUSVShape(100, dblY, dblX, dblYaw)
            plt.plot(matShape[0],matShape[1],color=strColor)
            plt.text(dblX,dblY,str(strID+"-Done"),color='g')


    def visualize(self):
        plt.clf()

        if self.lstAxisRange != None:
            plt.xlim(self.lstAxisRange[0],self.lstAxisRange[2])
            plt.ylim(self.lstAxisRange[1], self.lstAxisRange[3])
        if self.blnBackgroundImage == True:
            plt.imshow(self.objImage,extent=(self.lstAxisRange[0],self.lstAxisRange[2],\
                                             self.lstAxisRange[1],self.lstAxisRange[3]))

        lstVisualizeLog = []
        lstFireLog = []
        for lstLogInstance in self.lstLog:
            if lstLogInstance[0] >= self.dblCurrentTimestep and \
                    lstLogInstance[0] < self.dblCurrentTimestep+self.dblInterval:
                if lstLogInstance[2] == 'Maneuver':
                    lstVisualizeLog.append(lstLogInstance)
                if lstLogInstance[2] == 'Fire':
                    lstFireLog.append(lstLogInstance)

        # if self.dblCurrentTimestep == 0.0:
        self.visualizeWaypoint()
        self.visualizeEndState(self.dblCurrentTimestep)
        dicLastManeuverByID = {}
        for lstLogLine in lstVisualizeLog:
            if lstLogLine[1] not in dicLastManeuverByID.keys():
                dicLastManeuverByID[lstLogLine[1]] = lstLogLine
            else:
                if dicLastManeuverByID[lstLogLine[1]][0] < lstLogLine[0]:
                    dicLastManeuverByID[lstLogLine[1]] = lstLogLine

        for objKey in dicLastManeuverByID.keys():
            self.visualizeManeuver(dicLastManeuverByID[objKey])

        for lstLogInstance in lstFireLog:
            self.visualizeFire(lstLogInstance)

        self.dblCurrentTimestep = self.dblCurrentTimestep + self.dblInterval

        left, right = plt.xlim()
        top, bottom = plt.ylim()
        plt.text(left,top, \
                 "Log : "+str(self.strLogFile)+\
                 ", Timestep : "+str(self.dblCurrentTimestep))

        plt.xlabel("Position E")
        plt.ylabel("Position N")
        plt.grid(b=True,which="both",axis="both")
        if self.dblCurrentTimestep > self.dblMaxTime:
            plt.ioff()
            plt.show()
        else:
            # plt.ion()
            # plt.show(block=False)
            # plt.pause(0.01)
            self.pause(0.01)

    def animate(self):
        while True:
            if self.dblCurrentTimestep > self.dblMaxTime:
                break
            else:
                self.visualize()

    def pause(self,interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return

if __name__ == '__main__':
    objDisplay = LogDisplay('log.csv','./MapData/mission1.png',\
                            './MapData/occupancy_map.dat',100)
    objDisplay.animate()