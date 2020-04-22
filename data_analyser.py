# Import the pandas library.
import pandas
import matplotlib.pyplot as plt
import numpy as np

# < dip: right fingers >
def getDipRightMotion():
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
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    return dm, db, dm1, db1, dm2, db2, dm3, db3

# < dip: right fingers >
def getDipRightDistance():
    data = pandas.read_csv("dipRightDistance.csv", header=None, dtype=str)
    data.columns = ['rw2lw', 'rw2le', 'lw2re']
    rwlw = data['rw2lw'].astype(float)
    rwle = data['rw2le'].astype(float)
    lwre = data['lw2re'].astype(float)
    m, b = np.polyfit(rwlw, rwle, 1)
    m1, b1 = np.polyfit(rwlw, lwre, 1)
    return m, b, m1, b1

# < dip: right fingers >
def getDipRightOptical():
    data = pandas.read_csv("dipRightOptical.csv", header=None, dtype=str)
    data.columns = ['lw_mag', 'lw_ang', 'rw_mag', 'rw_ang', 'le_mag', 'le_ang', 're_mag', 're_ang']
    dx = data['rw_mag'].astype(float)
    dy = data['rw_ang'].astype(float)
    dx1 = data['lw_mag'].astype(float)
    dy1 = data['lw_ang'].astype(float)
    dx2 = data['le_mag'].astype(float)
    dy2 = data['le_ang'].astype(float)
    dx3 = data['re_mag'].astype(float)
    dy3 = data['re_ang'].astype(float)
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    return dm, db, dm1, db1, dm2, db2, dm3, db3

# < wash: right arm>
def getWashRightMotion():
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
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    return dm, db, dm1, db1, dm2, db2, dm3, db3

# < wash: right arm>
def getWashRightDistance():
    data = pandas.read_csv("washRightDistance.csv", header=None, dtype=str)
    data.columns = ['rw2lw', 'rw2le', 'lw2re']
    rwlw = data['rw2lw'].astype(float)
    rwle = data['rw2le'].astype(float)
    lwre = data['lw2re'].astype(float)
    m, b = np.polyfit(rwlw, rwle, 1)
    m1, b1 = np.polyfit(rwlw, lwre, 1)
    return m, b, m1, b1

def getWashRightOptical():
    data = pandas.read_csv("washRightOptical.csv", header=None, dtype=str)
    data.columns = ['lw_mag', 'lw_ang', 'rw_mag', 'rw_ang', 'le_mag', 'le_ang', 're_mag', 're_ang']
    dx = data['rw_mag'].astype(float)
    dy = data['rw_ang'].astype(float)
    dx1 = data['lw_mag'].astype(float)
    dy1 = data['lw_ang'].astype(float)
    dx2 = data['le_mag'].astype(float)
    dy2 = data['le_ang'].astype(float)
    dx3 = data['re_mag'].astype(float)
    dy3 = data['re_ang'].astype(float)
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    return dm, db, dm1, db1, dm2, db2, dm3, db3

# < dip: left fingers >
def getDipLeftMotion():
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
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    return dm, db, dm1, db1, dm2, db2, dm3, db3

def getDipLeftDistance():
    data = pandas.read_csv("dipLeftDistance.csv", header=None, dtype=str)
    data.columns = ['rw2lw', 'rw2le', 'lw2re']
    rwlw = data['rw2lw'].astype(float)
    rwle = data['rw2le'].astype(float)
    lwre = data['lw2re'].astype(float)
    m, b = np.polyfit(rwlw, rwle, 1)
    m1, b1 = np.polyfit(rwlw, lwre, 1)
    return m, b, m1, b1

def getDipLeftOptical():
    data = pandas.read_csv("dipLeftOptical.csv", header=None, dtype=str)
    data.columns = ['lw_mag', 'lw_ang', 'rw_mag', 'rw_ang', 'le_mag', 'le_ang', 're_mag', 're_ang']
    dx = data['rw_mag'].astype(float)
    dy = data['rw_ang'].astype(float)
    dx1 = data['lw_mag'].astype(float)
    dy1 = data['lw_ang'].astype(float)
    dx2 = data['le_mag'].astype(float)
    dy2 = data['le_ang'].astype(float)
    dx3 = data['re_mag'].astype(float)
    dy3 = data['re_ang'].astype(float)
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    return dm, db, dm1, db1, dm2, db2, dm3, db3

# < wash: left arm >
def getWashLeftMotion():
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
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    return dm, db, dm1, db1, dm2, db2, dm3, db3

def getWashLeftDistance():
    data = pandas.read_csv("washLeftDistance.csv", header=None, dtype=str)
    data.columns = ['rw2lw', 'rw2le', 'lw2re']
    rwlw = data['rw2lw'].astype(float)
    rwle = data['rw2le'].astype(float)
    lwre = data['lw2re'].astype(float)
    m, b = np.polyfit(rwlw, rwle, 1)
    m1, b1 = np.polyfit(rwlw, lwre, 1)
    return m, b, m1, b1

def getWashLeftOptical():
    data = pandas.read_csv("washLeftOptical.csv", header=None, dtype=str)
    data.columns = ['lw_mag', 'lw_ang', 'rw_mag', 'rw_ang', 'le_mag', 'le_ang', 're_mag', 're_ang']
    dx = data['rw_mag'].astype(float)
    dy = data['rw_ang'].astype(float)
    dx1 = data['lw_mag'].astype(float)
    dy1 = data['lw_ang'].astype(float)
    dx2 = data['le_mag'].astype(float)
    dy2 = data['le_ang'].astype(float)
    dx3 = data['re_mag'].astype(float)
    dy3 = data['re_ang'].astype(float)
    dm, db = np.polyfit(dx, dy, 1)
    dm1, db1 = np.polyfit(dx1, dy1, 1)
    dm2, db2 = np.polyfit(dx2, dy2, 1)
    dm3, db3 = np.polyfit(dx3, dy3, 1)
    return dm, db, dm1, db1, dm2, db2, dm3, db3

def getMotionScore(rw, lw, le, re):

    m, b, m1, b1, m2, b2, m3, b3 = getDipRightMotion()
    rw_res = m*rw.x + b - rw.y
    lw_res = m1*lw.x + b1 - lw.y
    le_res = m2*le.x + b2 - le.y
    re_res = m3*re.x + b3 - re.y
    dipRight = abs(rw_res) + abs(lw_res) + abs(le_res) + abs(re_res)

    m, b, m1, b1, m2, b2, m3, b3 = getWashRightMotion()
    rw_res = m*rw.x + b - rw.y
    lw_res = m1*lw.x + b1 - lw.y
    le_res = m2*le.x + b2 - le.y
    re_res = m3*re.x + b3 - re.y
    washRight = abs(rw_res) + abs(lw_res) + abs(le_res) + abs(re_res)

    m, b, m1, b1, m2, b2, m3, b3 = getDipLeftMotion()
    rw_res = m*rw.x + b - rw.y
    lw_res = m1*lw.x + b1 - lw.y
    le_res = m2*le.x + b2 - le.y
    re_res = m3*re.x + b3 - re.y
    dipLeft = abs(rw_res) + abs(lw_res) + abs(le_res) + abs(re_res)

    m, b, m1, b1, m2, b2, m3, b3 = getWashLeftMotion()
    rw_res = m*rw.x + b - rw.y
    lw_res = m1*lw.x + b1 - lw.y
    le_res = m2*le.x + b2 - le.y
    re_res = m3*re.x + b3 - re.y
    washLeft = abs(rw_res) + abs(lw_res) + abs(le_res) + abs(re_res)

    return dipRight, washRight, dipLeft, washLeft

def getDistanceScore(rw2lw, rw2le, lw2re):

    m, b, m1, b1 = getDipRightDistance()
    res1 = m*rw2lw + b - rw2le
    res2 = m1*rw2lw + b1 - lw2re
    dipRight = abs(res1) + abs(res2)

    m, b, m1, b1 = getWashRightDistance()
    res1 = m*rw2lw + b - rw2le
    res2 = m1*rw2lw + b1 - lw2re
    washRight = abs(res1) + abs(res2)

    m, b, m1, b1 = getDipLeftDistance()
    res1 = m*rw2lw + b - rw2le
    res2 = m1*rw2lw + b1 - lw2re
    dipLeft = abs(res1) + abs(res2)

    m, b, m1, b1 = getWashLeftDistance()
    res1 = m*rw2lw + b - rw2le
    res2 = m1*rw2lw + b1 - lw2re
    washLeft = abs(res1) + abs(res2)

    return dipRight, washRight, dipLeft, washLeft

# Optical Dataset: Not Used
def getOpticalScore(rw_mag, rw_ang, lw_mag, lw_ang, le_mag, le_ang, re_mag, re_ang):

    m, b, m1, b1, m2, b2, m3, b3 = getDipRightOptical()
    rw_res = m*rw_mag + b - rw_ang
    lw_res = m1*lw_mag + b1 - lw_ang
    le_res = m2*le_mag + b2 - le_ang
    re_res = m3*re_mag + b3 - re_ang
    dipRight = abs(rw_res) + abs(lw_res) + abs(le_res) + abs(re_res)

    m, b, m1, b1, m2, b2, m3, b3 = getWashRightOptical()
    rw_res = m*rw_mag + b - rw_ang
    lw_res = m1*lw_mag + b1 - lw_ang
    le_res = m2*le_mag + b2 - le_ang
    re_res = m3*re_mag + b3 - re_ang
    washRight = abs(rw_res) + abs(lw_res) + abs(le_res) + abs(re_res)

    m, b, m1, b1, m2, b2, m3, b3 = getDipLeftOptical()
    rw_res = m*rw_mag + b - rw_ang
    lw_res = m1*lw_mag + b1 - lw_ang
    le_res = m2*le_mag + b2 - le_ang
    re_res = m3*re_mag + b3 - re_ang
    dipLeft = abs(rw_res) + abs(lw_res) + abs(le_res) + abs(re_res)

    m, b, m1, b1, m2, b2, m3, b3 = getWashLeftOptical()
    rw_res = m*rw_mag + b - rw_ang
    lw_res = m1*lw_mag + b1 - lw_ang
    le_res = m2*le_mag + b2 - le_ang
    re_res = m3*re_mag + b3 - re_ang
    washLeft = abs(rw_res) + abs(lw_res) + abs(le_res) + abs(re_res)

    return dipRight, washRight, dipLeft, washLeft
