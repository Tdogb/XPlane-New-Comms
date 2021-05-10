import xpc
import time, csv
import numpy as np
import matplotlib.pyplot as plt

dS = []
controls = []
fakeTimes = []

posi_raw = []
controls_raw = []


SET_NUM_DATAPOINTS = 1000
STARTING_INDEX = 30

def monitor():
    with xpc.XPlaneConnect() as client:
        posi = client.getPOSI()
        posi_temp = list(posi)
        posi_temp[2] += 500
        client.sendPOSI(posi_temp[0:len(posi_temp)-9])
        client.sendPOSI(posi_temp[0:len(posi_temp)-9], 1)
        time.sleep(0.02)
        i = 0
        while len(posi_raw) < SET_NUM_DATAPOINTS:
            if posi != []:
                posi_raw.append(list(client.getPOSI()))
                controls_raw.append(list(client.getCTRL()))
                # fakeTimes.append(i)
                # i = i+1
                # plt.gca().lines[0].set_xdata(posi_raw)
                # plt.gca().lines[0].set_ydata(y)
                # plt.gca().relim()
                # plt.gca().autoscale_view()
                # plt.pause(0.02)
            time.sleep(0.02)

def graph():
    pass
'''
Overall NN:
Input:
- Controls
- Current velocities and angles
    - Roll, pitch, rollv, pitchv, heading velocity
- Not elevation or heading since we are trying to control these
- Not the timestep because that should stay constant, timestep only used in dS calculations

Output:
- Change in velocities, change in heading, change in elevation
- Needs to be normalized with the timestep to be unitless

Input
- Velocities
- 
- Change in elevation only
- Change in heading only
'''
def writeInputStates():
    indexOffset = 0 #Used to correct for indicies, shouldn't be changed after this is found
    with open('XPlane-Inputs.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for i in range(STARTING_INDEX, len(posi_raw)):
            controls = controls_raw[i - 1]
            elevation_ = posi_raw[i - 1][2 + indexOffset]
            roll_ = posi_raw[i - 1][3 + indexOffset]
            pitch_ = posi_raw[i - 1][4 + indexOffset]
            heading_ = posi_raw[i - 1][5 + indexOffset]
            vroll_ = posi_raw[i - 1][6 + indexOffset]
            vpitch_ = posi_raw[i - 1][7 + indexOffset]
            vheading_ = posi_raw[i - 1][8 + indexOffset]
            vx_ = posi_raw[i - 1][9 + indexOffset]
            vy_ = posi_raw[i - 1][10 + indexOffset]
            vz_ = posi_raw[i - 1][11 + indexOffset]
            time = posi_raw[i - 1][15 + indexOffset]
            row = [controls[0], controls[1], controls[2], controls[3], roll_, pitch_, vroll_, vpitch_, vheading_, vx_, vy_, vz_]
            writer.writerow(row)
            
def writeDesiredOutputStates():
    indexOffset = 0 #Used to correct for indicies, shouldn't be changed after this is found
    with open('XPlane-Outputs.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for i in range(STARTING_INDEX, len(posi_raw)):
            #NOTE THAT I IS AT STARTING_INDEX + 1
            elevation_ = posi_raw[i][2 + indexOffset]
            roll_ = posi_raw[i][3 + indexOffset]
            pitch_ = posi_raw[i][4 + indexOffset]
            heading_ = posi_raw[i][5 + indexOffset]
            vroll_ = posi_raw[i][6 + indexOffset]
            vpitch_ = posi_raw[i][7 + indexOffset]
            vheading_ = posi_raw[i][8 + indexOffset]
            vx_ = posi_raw[i][9 + indexOffset]
            vy_ = posi_raw[i][10 + indexOffset]
            vz_ = posi_raw[i][11 + indexOffset]
            time = posi_raw[i][15 + indexOffset]

            elevation_prev = posi_raw[i - 1][2 + indexOffset]
            roll_prev = posi_raw[i - 1][3 + indexOffset]
            pitch_prev = posi_raw[i - 1][4 + indexOffset]
            heading_prev = posi_raw[i - 1][5 + indexOffset]
            vroll_prev = posi_raw[i - 1][6 + indexOffset]
            vpitch_prev = posi_raw[i - 1][7 + indexOffset]
            vheading_prev = posi_raw[i - 1][8 + indexOffset]
            vx_prev = posi_raw[i - 1][9 + indexOffset]
            vy_prev = posi_raw[i - 1][10 + indexOffset]
            vz_prev = posi_raw[i - 1][11 + indexOffset]
            timeprev = posi_raw[i - 1][15 + indexOffset]
            row = [roll_-roll_prev, pitch_-pitch_prev, vroll_-vroll_prev, vpitch_-vpitch_prev, vheading_-vheading_prev, vx_-vx_prev, vy_-vy_prev, vz_-vz_prev, heading_-heading_prev, elevation_-elevation_prev]
            for b in range(0, len(row)):
                row[b] = row[b] / (time - timeprev)
            writer.writerow(row)
monitor()
writeInputStates()
writeDesiredOutputStates()