import numpy as np

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    for i in range(1,len(xdata)):

        line.axes.annotate('',
            xytext=(xdata[i-1], ydata[i-1]),
            xy=(xdata[i], ydata[i]),
            arrowprops=dict(arrowstyle="->", color=color),
            size=size
        )

def drawCircle(plt,X,Y,radius,color,strDashPattern):
    theta = np.linspace(0, 2 * np.pi, 100)

    a = radius * np.cos(theta) + X
    b = radius * np.sin(theta) + Y

    plt.plot(a,b,linestyle=strDashPattern,color=color)