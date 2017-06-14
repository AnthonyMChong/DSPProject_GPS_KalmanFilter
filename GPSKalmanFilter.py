import math
import csv
import numpy as np
import matplotlib.pyplot
import matplotlib.patches as mpatches

'''
Created by Anthony Chong for UCSC DSP class
Objective: create a kalman filter that can improve GPS accuracy given constant velocity and known initial state
using scalar estimate of 0.0000111 * m = degrees in lat/long
'''
# returns difference in seconds between GPS time stamp raw form and the previous second value
def TimeStamptoSec (TimeString):
    colonString = TimeString.split(" ")
    TimeSecondString = colonString[0].split(":")
    TimeSeconds = float(TimeSecondString[0])*360 + float(TimeSecondString[1])*60 + float(TimeSecondString[2])
    return (TimeSeconds)

#takes in velocity to output change in lat and long
def VelToGPSavg (inVelocity ,startingLat, startingLong ):
    with open("ConversionData") as Datafile:
        for line in Datafile:
            datalist = csv.reader(Datafile)
            datalist = list(datalist)
        dataradiuslist = []
        latdiff = []
        longdiff = []
        for n in range(0, len(datalist)):
            if float(datalist[n][3]) == 0.0:    # run these values
                dataradius = float(datalist[n][4])  # where data was taken in radius
                dataradiuslist.append(dataradius)
                currentlat = float (datalist[n][6])
                currentlong = float (datalist[n][7])
                latdiff.append((currentlat-startingLat)/dataradius)
                longdiff.append((currentlong-startingLong)/dataradius)
        latdiffpermeter = sum(latdiff)/len(latdiff)
        longdiffpermeter = sum(longdiff) / len(longdiff)
        return latdiffpermeter*inVelocity,longdiffpermeter*inVelocity

#we must define our physical model, this is to predict the user's position
MetToDeg = 0.0000111
userVelocity = 0.9        # velocity in meters per second, this value is the average walking speed
startingLat = 36.9999456              # our starting Latitude ground truth 36.9999456, -122.06234664
startingLong = -122.06234664             # our starting Longitude ground truth
latVelocity,longVelocity = VelToGPSavg(userVelocity, startingLat, startingLong) #velocity to deg of lat long
LongUncertainty = 0.5*MetToDeg # This represents a 3m uncertainty in our starting long
LatUncertainty = 0.5*MetToDeg # This represents a 3m uncertainty in our starting lat
LatVelocityUncertainty = latVelocity/10  # setting our uncertainty of initial velocity
LongVelocityUncertainty = longVelocity/10
FilteredLongitude = []
FilteredLatitude = []

# now reading our sensor data, this is contained in a CSV with measurements:
#time, lat, long, accuracy.


#####################################################################################################################
#Step 1: create initial predicted state/ physics model
#####################################################################################################################
#create the "A" matrix, this allows us to predict the state we are on, since we on t = 0 at start:
Amatrix = np.matrix([[1, 0],
                     [0, 1]])
#create a state matrix , this will be a 2x1 matrix including position and velocity in x OR y
Statematrix = np.matrix([[startingLong],
                         [longVelocity]])
StatematrixLat = np.matrix([[startingLat],
                         [latVelocity]])
# We know we do not change walking speed, the Predicted state is only vel and positon
PredictedState = Amatrix*Statematrix
PredictedStateLat = Amatrix*StatematrixLat

#####################################################################################################################
#Step 2: create initial process covariance Matrix
#####################################################################################################################
# This matrix holds the accuracy parameters of our initial estimate, the lower these values are, the more confident
InitialCovMatrix = np.matrix([[LongUncertainty**2 , LongUncertainty*LongVelocityUncertainty],
                              [LongUncertainty*LongVelocityUncertainty, LongVelocityUncertainty**2]])
PreviousCovMatrix = InitialCovMatrix
InitialCovMatrixLat = np.matrix([[LatUncertainty**2 , LatUncertainty*LatVelocityUncertainty],
                              [LatUncertainty*LatVelocityUncertainty, LatVelocityUncertainty**2]])
PreviousCovMatrixLat = InitialCovMatrixLat

with open("UnfilteredGPS") as Datafile:
    for line in Datafile:
        unfiltereddatalist= csv.reader(Datafile)
        unfiltereddatalist = list(unfiltereddatalist)


#####################################################################################################################
#Step 3: Predicted process covariance Matrix
#####################################################################################################################
#print (Amatrix)
#print (PreviousCovMatrix)
#print (Amatrix.transpose(1,0))

PredictedProcessCovMat =(Amatrix * PreviousCovMatrix)* Amatrix.transpose()
PreviousCovMatrix = PredictedProcessCovMat

PredictedProcessCovMatLat =(Amatrix * PreviousCovMatrixLat)* Amatrix.transpose()
PreviousCovMatrixLat = PredictedProcessCovMatLat


# starting first point
PreviousTime = TimeStamptoSec(unfiltereddatalist[0][0])
previousLong = float(startingLong)
previousLat = float(startingLat)
previousVelLong = float(longVelocity)
print ("VelLong "+str(previousVelLong))
previousVelLat = float(latVelocity)
print ("VelLat "+str(previousVelLat))
for n in range(0, unfiltereddatalist.__len__()):
    changeInTime = TimeStamptoSec(unfiltereddatalist[n][0]) - PreviousTime #used to determine Velocity
    PreviousTime = TimeStamptoSec(unfiltereddatalist[n][0])                 #saves for next change in time

    observedLong = float(unfiltereddatalist[n][2])
    observedLat = float(unfiltereddatalist[n][1])
    if changeInTime == 0:
        observedVelLong = longVelocity
        observedVelLat = latVelocity
    else:
        observedVelLong = (observedLong - previousLong) / changeInTime
        observedVelLat =(observedLat - previousLat) / changeInTime

    #####################################################################################################################
    #Step 4: Calculating the kalman gain
    #####################################################################################################################
    ## PredictedProcessCovMat
    ## H format changing matrix
    ## R observation error matrix
    ## KalmanGain
    Hmatrix = np.matrix([[1, 0],
                         [0, 1]])

    Rmatrix = np.matrix([[( previousLong - observedLong)**2, 0],
                         [0,(previousVelLong -observedVelLong)**2]])
    RmatrixLat = np.matrix([[(previousLat - observedLat) ** 2, 0],
                         [0, (previousVelLat - observedVelLat) ** 2]])

    print("\n\n Rmatrix \n" + str(Rmatrix))

    print("\n\n RmatrixLat \n" + str(RmatrixLat))

    KalmanGain = (PredictedProcessCovMat * Hmatrix.transpose())* (Hmatrix*PredictedProcessCovMat*Hmatrix.transpose()+Rmatrix)**-1
    KalmanGainLat = (PredictedProcessCovMatLat * Hmatrix.transpose())* (Hmatrix*PredictedProcessCovMatLat*Hmatrix.transpose()+RmatrixLat)**-1
    print ("\n\n Kalman Gain\n" + str(KalmanGain))
    print("\n\n Kalman Gain Lat\n" + str(KalmanGainLat))
    print(" PredictedProcessCovMatLat\n" + str(PredictedProcessCovMatLat))
    print(" RmatrixLat\n" + str(RmatrixLat))
    print(" PredictedStateLat\n" + str(PredictedStateLat) + "\n" +str(observedVelLat) )
    #####################################################################################################################
    #Step 5: New observation
    #####################################################################################################################
    # new observation
    SensorDataMatrix = np.matrix([[observedLong],
                                  [observedVelLong]])
    SensorDataMatrixLat = np.matrix([[observedLat],
                                  [observedVelLat]])
    Cmatrix = np.matrix([[1, 0],
                         [0, 1]])

    NewObeservation =  Cmatrix * SensorDataMatrix
    NewObeservationLat = Cmatrix * SensorDataMatrixLat
    print ("\n \n New observation\n"+str(NewObeservation))
    print("\n \n New observation Lat:\n" + str(NewObeservationLat))

    #####################################################################################################################
    #Step 6: Current State Estimation
    #####################################################################################################################
    # new current state
    # Kalman gain
    # H matrix
    # Previoust state estimate
    # Sensor Reading
    OldPredictedState = PredictedState
    PredictedState = OldPredictedState + KalmanGain*(NewObeservation - Hmatrix*OldPredictedState)
    previousLong = PredictedState.__array__()[0][0]
    previousVelLong = PredictedState.__array__()[1][0]

    print(" Previous Long: " + str(previousLong))
    print(" Previous Vel: " + str(previousVelLong) + "\n")
    print("\n \n Predicted State: \n "+ str(PredictedState))

    OldPredictedStateLat = PredictedStateLat
    PredictedStateLat = OldPredictedStateLat + KalmanGainLat*(NewObeservationLat - Hmatrix*OldPredictedStateLat)
    previousLat = PredictedStateLat.__array__()[0][0]
    previousVelLat = PredictedStateLat.__array__()[1][0]

    print(" Previous Lat: " + str(previousLat))
    print(" Previous VelLat: " + str(previousVelLat) + "\n")
    print("\n \n Predicted State Lat: \n " + str(PredictedStateLat))

    FilteredLatitude.append(previousLat)
    FilteredLongitude.append(previousLong)
    '''
    graph out points here and output to the filtered file
    '''

    #####################################################################################################################
    #Step 7: Updating the Process Covariance Matrix
    #####################################################################################################################
    # Getting new covariance matrix with kalman gain and previous state estimate
    Imatrix = np.matrix([[1, 0],
                         [0, 1]])
    PredictedProcessCovMat = (Imatrix - KalmanGain*Hmatrix)*PreviousCovMatrix
    print("\n\n New CovMatrix\n"+ str(PredictedProcessCovMat))
    PredictedProcessCovMatLat = (Imatrix - KalmanGainLat * Hmatrix) * PreviousCovMatrixLat
    print("\n\n New CovMatrixLat\n" + str(PredictedProcessCovMatLat))

    #####################################################################################################################
    #Step 8: new becomes old
    #####################################################################################################################
    # reStep 1: create initial predicted state/ physics model
    Amatrix = np.matrix([[1, changeInTime],
                         [0, 1]])
    print("\n\n AMatrix \n" + str(Amatrix))
    Statematrix = PredictedState    # predicted state will contain lat/long and its respective velocity
    PredictedState = Amatrix * Statematrix
    print("\n\n PredictedStateMatrix \n" + str(PredictedState))
    StatematrixLat = PredictedStateLat  # predicted state will contain lat/long and its respective velocity
    PredictedStateLat = Amatrix * StatematrixLat
    print("\n\n PredictedStateMatrixLat \n" + str(PredictedStateLat))

###### Mapping out filtered points
FilteredOutput = open("FilteredGPS.txt",'w',newline='')
FilteredOutput.write("Lat , Long \n")
for coord in range (0,len(FilteredLongitude)):
    instring = str(FilteredLatitude[coord]) + "," + str(FilteredLongitude[coord])+"\n"
    FilteredOutput.write(instring)
FilteredOutput.close()

img = matplotlib.pyplot.imread("GPS_DSP_MAP.JPG")
fig, ax = matplotlib.pyplot.subplots()
figLongLow = -122.062520
figLongHigh = -122.060687
figLatLow = 36.999780
figLatHigh = 37.000590
ax.imshow(img, extent=[figLongLow, figLongHigh, figLatLow ,figLatHigh ])
#Bottomleft 36.999780,-122.062520
#TopRight  37.000590, -122.060687

actualoriginlong = -122.06234664
actualoriginlat = 36.9999456

findstatLat = []
findstatLong = []

ax = fig.gca()
previousLat = 36.9999456
onlyEstLat = previousLat
previousLong = -122.06234664
onlyEstLong = previousLong

for n in range(0,len(unfiltereddatalist)):
    GPSestimateLat = float(unfiltereddatalist[n][1])  # Enter index of predicted X here
    GPSestimateLong = float(unfiltereddatalist[n][2])  # Enter index of predicted Y here
    matplotlib.pyplot.plot(GPSestimateLong, GPSestimateLat, 'wo')

for n in range(0,len(FilteredLatitude)):
    GPSestimateLat = float(FilteredLatitude[n])  # Enter index of predicted X here
    GPSestimateLong = float(FilteredLongitude[n])  # Enter index of predicted Y here
    matplotlib.pyplot.plot(GPSestimateLong, GPSestimateLat, 'mo')
    #matplotlib.pyplot.plot([previousLong, GPSestimateLong], [previousLat, GPSestimateLat], 'g-', lw=2)
    previousLat = GPSestimateLat
    previousLong = GPSestimateLong
    onlyEstLat += latVelocity*3
    onlyEstLong += longVelocity*3
matplotlib.pyplot.plot([-122.06234664, onlyEstLong], [36.9999456, onlyEstLat], 'y-', lw=2)


GraphWhiteSpace = 0.00002
matplotlib.pyplot.ylim(figLatLow ,figLatHigh)
matplotlib.pyplot.xlim(figLongLow ,figLongHigh)
red_patch = mpatches.Patch(color='red', label='GPS Measurements')
matplotlib.pyplot.legend(handles=[red_patch])

matplotlib.pyplot.show()

