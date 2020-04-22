# STATE MACHINE

DIP_R = 'dip right'
DIP_L = 'dip left'
WASH_R = 'wash right'
WASH_L = 'wash left'
UNCLASSIFIED = ''

def dipRightState(current):
    if (current == UNCLASSIFIED):
        return True
    if (current == WASH_R):
        return True
    elif (current == DIP_R):
        return True
    return False

def dipLeftState(current):
    if (current == WASH_L):
        return True
    elif (current == DIP_L):
        return True
    return False

def washRightState(current):
    if (current == DIP_L):
        return True
    elif (current == WASH_R):
        return True
    return False

def washLeftState(current):
    if (current == WASH_L):
        return True
    return False

def acceptedState(current, previous):
    if (previous == UNCLASSIFIED):
        return True
    if (previous == DIP_R):
        return dipRightState(current)
    if (previous == WASH_R):
        return washRightState(current)
    if (previous == DIP_L):
        return dipLeftState(current)
    if (previous == WASH_L):
        return washLeftState(current)
