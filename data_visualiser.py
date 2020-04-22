# Import the pandas library.
import pandas
import matplotlib.pyplot as plt
import numpy as np

def visualiseDipRightLocation():
    # read data
    data = pandas.read_csv("dipRight.csv", header=None, dtype=str)
    data.columns = ['lw_x', 'lw_y', 'rw_x', 'rw_y', 'le_x', 'le_y', 're_x', 're_y']
    dx = data['rw_x'].astype(float)
    dy = data['rw_y'].astype(float)
    dx1 = data['lw_x'].astype(float)
    dy1 = data['lw_y'].astype(float)
    dx2 = data['le_x'].astype(float)
    dy2 = data['le_y'].astype(float)
    dx3 = data['re_x'].astype(float)
    dy3 = data['re_y'].astype(float)
    # scatter plot of datapoints
    plt.scatter(dx, dy, c='r')
    plt.scatter(dx1, dy1, c='y')
    plt.scatter(dx2, dy2, c='b')
    plt.scatter(dx3, dy3, c='g')
    # line of best fit
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    plt.plot(dx, dm*dx + db, 'r')
    plt.plot(dx1, dm1*dx1 + db1, 'y')
    plt.plot(dx2, dm2*dx2 + db2, 'b')
    plt.plot(dx3, dm3*dx3 + db3, 'g')
    # display graph
    plt.gca().invert_yaxis()
    plt.show()

def visualiseDipRightDistance():
    # read data
    data = pandas.read_csv("dipRightDistance.csv", header=None, dtype=str)
    data.columns = ['rw2lw', 'rw2le', 'lw2re']
    dx = data['rw2lw'].astype(float)
    dy = data['rw2le'].astype(float)
    dy1 = data['lw2re'].astype(float)
    # scatter plot of datapoints
    plt.scatter(dx, dy, c='r')
    plt.scatter(dx, dy1, c='y')
    # line of best fit
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx, dy1, 1)
    plt.plot(dx, dm*dx + db, 'r')
    plt.plot(dx, dm1*dx + db1, 'y')
    # display graph
    plt.gca().invert_yaxis()
    plt.show()

def visualiseDipLeftLocation():
    # read data
    data = pandas.read_csv("dipLeft.csv", header=None, dtype=str)
    data.columns = ['lw_x', 'lw_y', 'rw_x', 'rw_y', 'le_x', 'le_y', 're_x', 're_y']
    dx = data['rw_x'].astype(float)
    dy = data['rw_y'].astype(float)
    dx1 = data['lw_x'].astype(float)
    dy1 = data['lw_y'].astype(float)
    dx2 = data['le_x'].astype(float)
    dy2 = data['le_y'].astype(float)
    dx3 = data['re_x'].astype(float)
    dy3 = data['re_y'].astype(float)
    # scatter plot of datapoints
    plt.scatter(dx, dy, c='r')
    plt.scatter(dx1, dy1, c='y')
    plt.scatter(dx2, dy2, c='b')
    plt.scatter(dx3, dy3, c='g')
    # line of best fit
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    plt.plot(dx, dm*dx + db, 'r')
    plt.plot(dx1, dm1*dx1 + db1, 'y')
    plt.plot(dx2, dm2*dx2 + db2, 'b')
    plt.plot(dx3, dm3*dx3 + db3, 'g')
    # display graph
    plt.gca().invert_yaxis()
    plt.show()

def visualiseDipLeftDistance():
    # read data
    data = pandas.read_csv("dipLeftDistance.csv", header=None, dtype=str)
    data.columns = ['rw2lw', 'rw2le', 'lw2re']
    dx = data['rw2lw'].astype(float)
    dy = data['rw2le'].astype(float)
    dy1 = data['lw2re'].astype(float)
    # scatter plot of datapoints
    plt.scatter(dx, dy, c='r')
    plt.scatter(dx, dy1, c='y')
    # line of best fit
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx, dy1, 1)
    plt.plot(dx, dm*dx + db, 'r')
    plt.plot(dx, dm1*dx + db1, 'y')
    # display graph
    plt.gca().invert_yaxis()
    plt.show()

def visualiseWashRightLocation():
    # read data
    data = pandas.read_csv("washRight.csv", header=None, dtype=str)
    data.columns = ['lw_x', 'lw_y', 'rw_x', 'rw_y', 'le_x', 'le_y', 're_x', 're_y']
    dx = data['rw_x'].astype(float)
    dy = data['rw_y'].astype(float)
    dx1 = data['lw_x'].astype(float)
    dy1 = data['lw_y'].astype(float)
    dx2 = data['le_x'].astype(float)
    dy2 = data['le_y'].astype(float)
    dx3 = data['re_x'].astype(float)
    dy3 = data['re_y'].astype(float)
    # scatter plot of datapoints
    plt.scatter(dx, dy, c='r')
    plt.scatter(dx1, dy1, c='y')
    plt.scatter(dx2, dy2, c='b')
    plt.scatter(dx3, dy3, c='g')
    # line of best fit
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    plt.plot(dx, dm*dx + db, 'r')
    plt.plot(dx1, dm1*dx1 + db1, 'y')
    plt.plot(dx2, dm2*dx2 + db2, 'b')
    plt.plot(dx3, dm3*dx3 + db3, 'g')
    # display graph
    plt.gca().invert_yaxis()
    plt.show()

def visualiseWashRightDistance():
    # read data
    data = pandas.read_csv("washRightDistance.csv", header=None, dtype=str)
    data.columns = ['rw2lw', 'rw2le', 'lw2re']
    dx = data['rw2lw'].astype(float)
    dy = data['rw2le'].astype(float)
    dy1 = data['lw2re'].astype(float)
    # scatter plot of datapoints
    plt.scatter(dx, dy, c='r')
    plt.scatter(dx, dy1, c='y')
    # line of best fit
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx, dy1, 1)
    plt.plot(dx, dm*dx + db, 'r')
    plt.plot(dx, dm1*dx + db1, 'y')
    # display graph
    plt.gca().invert_yaxis()
    plt.show()

def visualiseWashLeftLocation():
    # read data
    data = pandas.read_csv("washLeft.csv", header=None, dtype=str)
    data.columns = ['lw_x', 'lw_y', 'rw_x', 'rw_y', 'le_x', 'le_y', 're_x', 're_y']
    dx = data['rw_x'].astype(float)
    dy = data['rw_y'].astype(float)
    dx1 = data['lw_x'].astype(float)
    dy1 = data['lw_y'].astype(float)
    dx2 = data['le_x'].astype(float)
    dy2 = data['le_y'].astype(float)
    dx3 = data['re_x'].astype(float)
    dy3 = data['re_y'].astype(float)
    # scatter plot of datapoints
    plt.scatter(dx, dy, c='r')
    plt.scatter(dx1, dy1, c='y')
    plt.scatter(dx2, dy2, c='b')
    plt.scatter(dx3, dy3, c='g')
    # line of best fit
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    plt.plot(dx, dm*dx + db, 'r')
    plt.plot(dx1, dm1*dx1 + db1, 'y')
    plt.plot(dx2, dm2*dx2 + db2, 'b')
    plt.plot(dx3, dm3*dx3 + db3, 'g')
    # display graph
    plt.gca().invert_yaxis()
    plt.show()

def visualiseWashLeftDistance():
    # read data
    data = pandas.read_csv("washLeftDistance.csv", header=None, dtype=str)
    data.columns = ['rw2lw', 'rw2le', 'lw2re']
    dx = data['rw2lw'].astype(float)
    dy = data['rw2le'].astype(float)
    dy1 = data['lw2re'].astype(float)
    # scatter plot of datapoints
    plt.scatter(dx, dy, c='r')
    plt.scatter(dx, dy1, c='y')
    # line of best fit
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx, dy1, 1)
    plt.plot(dx, dm*dx + db, 'r')
    plt.plot(dx, dm1*dx + db1, 'y')
    # display graph
    plt.gca().invert_yaxis()
    plt.show()

visualiseDipRightLocation()
visualiseDipRightDistance()
visualiseDipLeftLocation()
visualiseDipLeftDistance()
visualiseWashRightLocation()
visualiseWashRightDistance()
visualiseWashLeftLocation()
visualiseWashLeftDistance()
