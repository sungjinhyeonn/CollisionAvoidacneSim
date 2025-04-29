import math
import csv
import datetime
import sys
import traceback
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np

class SIMDISLogger(object):

    def __init__(self, dblReferenceLatitude, dblReferenceLongitude, blnVisualize3D, \
                 intVisualize3DInterval,intHistPathLength, strShowAttitudeFighter, \
                 dblMinX,dblMinY,dblMinZ,dblMaxX,dblMaxY,dblMaxZ):
        """ generated source for class SIMDISLogger """

        try:
            #  xplane connection creation
            self.strShowAttitudeFighter = strShowAttitudeFighter
            self.intHistPathLength = intHistPathLength
            self.blnVisualize3D = blnVisualize3D
            self.intVisualize3DInterval = intVisualize3DInterval
            self.intID = 1
            self.dicIDbyName = {}
            self.dicIntervalbyName = {}
            self.dicStatebyName = {}
            #  log output stream creation
            dt = datetime.datetime.now()
            year = dt.year
            month = dt.month
            day = dt.day
            hour = dt.hour
            if hour < 10:
                hour = str(0) + str(hour)
            minute = dt.minute
            if minute < 10:
                minute = str(0) + str(minute)
            second = dt.second
            if second < 10:
                second = str(0) + str(second)
            if sys.version_info[0] == 3:
                SIMDISFile = open("SIMDIS-"+str(year)+"-"+str(month)+"-"+str(day)+"-"+str(hour)+"-"+str(minute)+"-"+str(second)+".asi","w")
            else:
                SIMDISFile = open("SIMDIS-" + str(year) + "-" + str(month) + "-" + str(day) + "-" + str(hour) + "-" + str(minute) + "-" + str(second) + ".asi", "w")
            self.f = SIMDISFile
        except Exception as e:
            print(e)
            print("!!! SIMDISLogger initialize failed")
            traceback.print_stack()
            sys.exit()

        """ generated source for method initialize """
        self.f.write("# SIMDIS Ascii Scenario Input (ASI) File Format\n")
        self.f.write("# Scenario Initialization\n")
        self.f.write("Version 5\n")
        self.f.write("RefLat " + str(dblReferenceLatitude) + '\n')
        self.f.write("RefLon " + str(dblReferenceLongitude) + '\n')
        self.f.write('CoordSystem LLA\n')
        self.f.write("DegreeAngles 1\n")
        self.f.write("# Platform Header\n")

        if self.blnVisualize3D == True:
            self.fig = plt.figure()
            self.axAdaptive = plt.subplot2grid((3,3),(0,0),colspan=2,rowspan=2,projection='3d')
            self.axAdaptive.title.set_text('Adaptive Scale from Quarter View')

            self.axFixedN = plt.subplot2grid((3,3),(2,0),colspan=1,rowspan=1,projection='3d')
            self.scaleFixed(self.axFixedN,dblMinX,dblMaxX,dblMinY,dblMaxY,dblMinZ,dblMaxZ)
            self.axFixedN.view_init(0, 0)
            self.axFixedN.title.set_text('Fixed Scale for Altitude')

            self.axFixedD = plt.subplot2grid((3,3),(2,1),colspan=1,rowspan=1,projection='3d')
            self.scaleFixed(self.axFixedD,dblMinX,dblMaxX,dblMinY,dblMaxY,dblMinZ,dblMaxZ)
            self.axFixedD.view_init(90, 90)
            self.axFixedD.title.set_text('Fixed Scale for N, E')

            self.axAttitude = plt.subplot2grid((3,3),(2,2),colspan=1,rowspan=1,projection='3d')
            self.axAttitude.title.set_text('Attitude of '+self.strShowAttitudeFighter)

            plt.tight_layout()

            self.dicPlotByName = {}

    def registerPlatform(self, strName, strType, strIcon):
        """ generated source for method registerPlatform """
        id = self.intID
        self.dicIDbyName[strName] = id
        self.dicIntervalbyName[strName] = 0
        self.dicStatebyName[strName] = []
        self.intID = self.intID + 1
        self.f.write("PlatformID " + str(id) + '\n')
        self.f.write("PlatformName " + str(id) + ' "' + strName + '"\n')
        self.f.write("PlatformType " + str(id) + ' "' + strType + '"\n')
        self.f.write("PlatformIcon " + str(id) + ' "' + strIcon + '"\n')
        self.f.write("# Platform Data\n")
        self.f.write("# PlatformData PlatformID Time Lat Lon Alt Yow Pitch Roll \n")

    def recordDead(self, strName):
        if self.blnVisualize3D == True:
            for i in range(len(self.dicPlotByName[strName])):
                if self.dicPlotByName[strName][i] != None:
                    self.dicPlotByName[strName][i].remove()
            self.dicStatebyName.pop(strName)

    def recordMovement(self, strName, dblTime, dblLatitude, dblLongitude, dblAltitude, dblYaw, dblPitch, dblRoll, \
                       dblPositionN, dblPositionE, dblPositionD):
        """ generated source for method recordMovement """
        RAD2DEG = 57.29578
        id = self.dicIDbyName[strName]
        row = "PlatformData " + str(id) + " " + str(dblTime) + " " + str(dblLatitude) + " " + str(dblLongitude) + " " + \
              str(dblAltitude) + " " + str(dblYaw * RAD2DEG) + " " + str(dblPitch * RAD2DEG) + " " + str(dblRoll * RAD2DEG) + "\n"
        self.f.write(row)

        self.dicStatebyName[strName].append([dblPositionN,dblPositionE,-dblPositionD,dblYaw,dblPitch,dblRoll])
        if self.blnVisualize3D == True:
            if self.dicIntervalbyName[strName] == 0:
                self.animate3DPlot(strName)
            self.dicIntervalbyName[strName] = self.dicIntervalbyName[strName] + 1
            if self.dicIntervalbyName[strName] >= self.intVisualize3DInterval:
                self.dicIntervalbyName[strName] = 0

    def animate3DPlot(self, strName):
        xs = []
        ys = []
        zs = []
        l = 10.0
        lstPrefix = ["Blue","Red"]
        lstColor = ['b','r']
        lstColorHist = ['c', 'm']
        strColor = None
        strColorHist = None
        for i in range(len(lstPrefix)):
            if strName.startswith(lstPrefix[i]) == True:
                strColor = lstColor[i]
                strColorHist = lstColorHist[i]

        lstState = self.dicStatebyName[strName][-1]

        xs = [lstState[0], \
              lstState[0] + l * math.cos(lstState[3]) * math.cos(lstState[4]), \
              lstState[0] + l / 2.0 * math.cos(math.pi/2.0-lstState[3]) * math.cos(lstState[5]), \
              lstState[0] - l / 2.0 * math.cos(math.pi/2.0-lstState[3]) * math.cos(lstState[5]), \
              lstState[0] + l * math.cos(lstState[3]) * math.cos(lstState[4]), \
              ]
        ys = [lstState[1], \
              lstState[1] + l * math.sin(lstState[3]) * math.cos(lstState[4]), \
              lstState[1] + l / 2.0 * math.sin(math.pi/2.0 - lstState[3]) * math.cos(lstState[5]), \
              lstState[1] - l / 2.0 * math.sin(math.pi/2.0 - lstState[3]) * math.cos(lstState[5]), \
              lstState[1] + l * math.sin(lstState[3]) * math.cos(lstState[4]), \
              ]
        zs = [lstState[2], \
              lstState[2] + l * math.sin(lstState[4]), \
              lstState[2] + l / 2.0 * math.sin(lstState[5]), \
              lstState[2] - l / 2.0 * math.sin(lstState[5]), \
              lstState[2] + l * math.sin(lstState[4]), \
              ]

        intHistStart = len(self.dicStatebyName[strName]) - self.intHistPathLength
        if intHistStart < 0:
            intHistStart = 0

        histxs = []
        histys = []
        histzs = []

        for i in range(intHistStart,len(self.dicStatebyName[strName])):
            histxs.append(self.dicStatebyName[strName][i][0])
            histys.append(self.dicStatebyName[strName][i][1])
            histzs.append(self.dicStatebyName[strName][i][2])

        if strName not in self.dicPlotByName.keys():
            objPlotAdaptive, = self.axAdaptive.plot(xs, ys, zs, c=strColor)
            objPlotHistAdaptive, = self.axAdaptive.plot(histxs, histys, histzs, c=strColorHist)
            objPlotFixedN, = self.axFixedN.plot(xs, ys, zs, c=strColor)
            objPlotFixedD, = self.axFixedD.plot(xs, ys, zs, c=strColor)
            if strName == self.strShowAttitudeFighter:
                objPlotAttitude, = self.axAttitude.plot(xs, ys, zs, c=strColor)
            else:
                objPlotAttitude = None

            self.dicPlotByName[strName] = [objPlotAdaptive,objPlotFixedN,objPlotFixedD,objPlotAttitude,objPlotHistAdaptive]

        else:
            objPlots = self.dicPlotByName[strName]
            xys = np.empty((2, len(xs)))
            for i in range(len(xs)):
                xys[0, i] = xs[i]
                xys[1, i] = ys[i]
            histxys = np.empty((2, len(histxs)))
            for i in range(len(histxs)):
                histxys[0, i] = histxs[i]
                histxys[1, i] = histys[i]
            for i in range(len(objPlots)-1):
                if objPlots[i] != None:
                    objPlots[i].set_data(xys)
                    objPlots[i].set_3d_properties(zs)
            objPlots[-1].set_data(histxys)
            objPlots[-1].set_3d_properties(histzs)
        if strName == self.strShowAttitudeFighter:
            self.scaleFixed(self.axAttitude,lstState[0]-20,lstState[0]+20, \
                        lstState[1]-20,lstState[1]+20,lstState[2]-20,lstState[2]+20)
        self.scaleAutomatic(self.axAdaptive)
        plt.ion()
        plt.show(block=False)
        plt.pause(0.01)

    def scaleFixed(self,ax,dblMinX=0,dblMaxX=3000,dblMinY=0,dblMaxY=3000,dblMinZ=0,dblMaxZ=20000):
        ax.set_xlim3d(dblMinX,dblMaxX)
        ax.set_ylim3d(dblMinY,dblMaxY)
        ax.set_zlim3d(dblMinZ,dblMaxZ)

    def scaleAutomatic(self,ax,dblTightness=0.5, dblMinPadding = 100):

        dblMinX = 9999999999
        dblMaxX = -9999999999
        dblMinY = 9999999999
        dblMaxY = -9999999999
        dblMinZ = 9999999999
        dblMaxZ = -9999999999

        for key in self.dicStatebyName.keys():
            if len(self.dicStatebyName[key]) != 0:
                if self.dicStatebyName[key][-1][0] < dblMinX:
                    dblMinX = self.dicStatebyName[key][-1][0]
                if self.dicStatebyName[key][-1][0] > dblMaxX:
                    dblMaxX = self.dicStatebyName[key][-1][0]
                if self.dicStatebyName[key][-1][1] < dblMinY:
                    dblMinY = self.dicStatebyName[key][-1][1]
                if self.dicStatebyName[key][-1][1] > dblMaxY:
                    dblMaxY = self.dicStatebyName[key][-1][1]
                if self.dicStatebyName[key][-1][2] < dblMinZ:
                    dblMinZ = self.dicStatebyName[key][-1][2]
                if self.dicStatebyName[key][-1][2] > dblMaxZ:
                    dblMaxZ = self.dicStatebyName[key][-1][2]

        dblPaddingX = (dblMaxX - dblMinX) * dblTightness + dblMinPadding
        dblPaddingY = (dblMaxY - dblMinY) * dblTightness + dblMinPadding
        dblPaddingZ = (dblMaxZ - dblMinZ) * dblTightness + dblMinPadding
        ax.set_xlim3d(dblMinX-dblPaddingX,dblMaxX+dblPaddingX)
        ax.set_ylim3d(dblMinY-dblPaddingY,dblMaxY+dblPaddingY)
        ax.set_zlim3d(dblMinZ-dblPaddingZ,dblMaxZ+dblPaddingZ)

