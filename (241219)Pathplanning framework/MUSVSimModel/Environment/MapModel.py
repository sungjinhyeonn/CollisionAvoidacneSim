import csv
import os
import numpy as np
import matplotlib.pyplot as plt

class GridEnvironment:

    def __init__(self,strGridDataFilename,dblPixelMeterScale):
        objFile = open(strGridDataFilename,'r')
        objReader = csv.reader(objFile,delimiter=' ')

        lstMatrix = []
        for lstLine in objReader:
            lstTemp = []
            for strTemp in lstLine:
                lstTemp.append(int(float(strTemp)))
            lstMatrix.append(lstTemp)

        self.matGrid = np.asarray(lstMatrix)
        self.dblPixelMeterScale = dblPixelMeterScale

    def isValidPositionNE(self,dblPositionN,dblPositionE):
        intGridN = self.matGrid.shape[0] - int( dblPositionN / self.dblPixelMeterScale )
        intGridE = int( dblPositionE / self.dblPixelMeterScale )

        if self.matGrid[intGridN][intGridE] == 1:
            return True
        else:
            return False

    def getShape(self):
        return self.matGrid.shape[1],self.matGrid.shape[0]

    def isValidGrid(self,intY,intX):
        if intY <0 or intY >= self.matGrid.shape[0]:
            return True
        if intX <0 or intX >= self.matGrid.shape[1]:
            return True

        if self.matGrid[intY][intX] == 1:
            return True
        else:
            return False

    def visualizeMap(self,intGridN=None,intGridE=None):
        if intGridN != None and intGridE != None:
            plt.plot([intGridE],[intGridN],marker="o",color="r")
        plt.imshow(self.matGrid)
        plt.show()

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    objGrid = GridEnvironment("../../MapData/occupancy_map.dat",100)
    objGrid.visualizeMap()