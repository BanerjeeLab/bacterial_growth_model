import numpy as np
from numpy.random import normal as gauss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd 

##############################################################################
# Define fitted parameters and distributions, methods
##############################################################################

#Noise amplitude for the length langevin equation
def nu_k(kbar):
    return 0.41236364*kbar+0.01392364 #0.025#0.03
#Noise amplitude for the width langevin equation
def eta_w(kbar):
    return 0.0478*kbar+0.0018

#variation of timeseries cell widths about the mean width 
def w_SD_fit(kbar):
    return 0.017

#mean value of lambda
def lambda_fit(kbar):
    return 19.213*kbar+.54602
#Cell mean width and population noise
def w0_fit(kbar):
    return 10.9515*kbar+0.309853
def w0_SD_fit(kbar):
    return 0.57063*kbar+0.0228199

# k, lambda Correlated multivariate gaussian parameters 
def K_fit(kbar):
    return 11.5789*kbar-1.89861
def K_SD_fit(kbar):
    return (62.4059*kbar**2-5.51351*kbar+0.198175)
def Lambda_fit(kbar):
    return 9.49035*kbar+0.928367
def Lambda_SD_fit(kbar):
    return (1.71782*kbar+0.147916)
def lambda_max_fit(kbar):
    return 46.2813*kbar+1.26034
def rho_LambdaK_fit(kbar):
    return -.678776*kbar-0.857904

#Adder model parameters
def Delta_fit(kbar):
    return 16.5201*kbar+0.338082
def Delta_SD_fit(kbar):
    return 4451.02*kbar**4+0.176088
def L0_fit(kbar):
    return Delta_fit(kbar)
def L0_SD_fit(kbar):
    return Delta_SD_fit(kbar)

#Division ratio
def r_fit(kbar):
    return 0.499925
def r_SD_fit(kbar):
    return 0.0338709

#Intergenerational correlation for cell mean width
def rho_inter_w0(kbar):
    return 2.93892*kbar+0.642575

#Use the correlated distrubution to draw values of k and lambda for a cell
def draw_noise(kbar):
    mean = [K_fit(kbar), Lambda_fit(kbar)]
    covOD = K_SD_fit(kbar)*Lambda_SD_fit(kbar)*rho_LambdaK_fit(kbar)
    cov = [[K_SD_fit(kbar)**2, covOD], [covOD, Lambda_SD_fit(kbar)**2]]
    KSim, LambdaSim = np.random.multivariate_normal(mean, cov, 1).T
    kSim = 10**KSim
    lambdaSim = lambda_max_fit(kbar)-LambdaSim**2
    return kSim[0], lambdaSim[0]

#Update length and width for the next timestep
def langevin_length_simple(L, meank, lambdamean, kbar, dt):
    weiner = gauss(0, dt**.5)
    return L+meank*(L-lambdamean)*dt+nu_k(kbar)*weiner
def langevin_width(w, w0current, kbar, dt):
    deltaw = w-w0current
    weiner = gauss(0, dt**.5)
    return w+(-kbar*deltaw)*dt+weiner*eta_w(kbar)

#Input cell state, output dt later
def update_state(L, w, kbar, kNew, lambdaNew, w0current, dt):
    L = langevin_length_simple(L, kNew, lambdaNew, kbar, dt)
    w = langevin_width(w, w0current, kbar, dt)
    return L, w

#Perform cell division and follow one daughter cell 
def cell_division(L, w, w0current, kbar):
    r = gauss(r_fit(kbar), r_SD_fit(kbar))
    L = L*r
    w0norm = rho_inter_w0(kbar)*(w0current-w0_fit(kbar))/w0_SD_fit(kbar)+gauss(0, (1-rho_inter_w0(kbar)**2)**.5)
    w0 = w0_SD_fit(kbar)*w0norm+w0_fit(kbar)
    w = gauss(w0, w_SD_fit(kbar))
    kNew, lambdaNew = draw_noise(kbar)
    return L, w, w0, kNew, lambdaNew

#L(t) and kappa(t) from our non-stochastic model
def modelLFit(t, A, k, lam):
    return A*np.exp(k*t)+lam
def modelkFit(t, A, k, lam):
    return A*k*np.exp(k*t)/(A*np.exp(k*t)+lam)

#[Input] Ribosomal growth rate, could change over time for nutrient shifts
def kbar_t(t):
    return .04#.07#0.0408131
#Value of alpha*R0 fit to data
def alphaFit(kbar):
    return -31.63636369*kbar+4.08363637 

##############################################################################
#Run Sample Simulation
##############################################################################

#Main method that outputs the timeseries of length and width
#printSwitchD is whether or not you want each division to be printed
def FULL_SIMULATION(generations, dt, printSwitchD):
    #Lists to keep track of various quantities for each generation
    tauRecord = []
    l0Record = []
    # initialize
    cellLengths = np.zeros(1)
    cellLengths[0] = np.random.lognormal(
        Delta_fit(kbar_t(0)), Delta_SD_fit(kbar_t(0)))
    cellWidths = np.zeros(1)
    cellWidths[0] = gauss(w0_fit(kbar_t(0)), w0_SD_fit(kbar_t(0)))
    cellMeank = np.zeros(1)
    cellMeanlambda = np.zeros(1)
    junk, junk2, junk3, cellMeank[0], cellMeanlambda[0] = cell_division(
        cellLengths[0], cellWidths[0], cellWidths[0], kbar_t(0))
    newCellTimestamps = np.zeros(1)
    # Index for number of timesteps undergone
    ndt = 0
    divisionNumber = 0
    divisionDelta = np.random.lognormal(
        Delta_fit(kbar_t(0)), Delta_SD_fit(kbar_t(0)))
    w0current = cellWidths[0]
    kNew = cellMeank[0]
    lambdaNew = cellMeanlambda[0]
    # Run simulation for pre-determined number of generations
    while divisionNumber < generations:
        # check if division occurs
        if cellLengths[ndt] > cellLengths[int(newCellTimestamps[divisionNumber])]+divisionDelta:
            # divide cell, **making sure that lambda is less than cell length**
            while True:
                LNew, wNew, w0currentNew, kNew, lambdaNew = cell_division(
                    cellLengths[ndt], cellWidths[ndt], w0current, kbar_t(ndt*dt))
                if LNew > lambdaNew+.2 and kNew>0 and lambdaNew>0:
                    w0current = w0currentNew
                    break
            cellLengths = np.append(cellLengths, LNew)
            cellWidths = np.append(cellWidths, wNew)
            cellMeank = np.append(cellMeank, kNew)
            cellMeanlambda = np.append(cellMeanlambda, lambdaNew)
            ndt += 1
            divisionNumber += 1
            # mark new initial cell length
            newCellTimestamps = np.append(newCellTimestamps, ndt)
            tau = dt*(newCellTimestamps[divisionNumber] -
                      newCellTimestamps[divisionNumber-1])
            tauRecord.append(tau)
            l0Record.append(LNew)
            # reset Delta for next division
            divisionDelta = np.random.lognormal(
                Delta_fit(kbar_t(ndt*dt)), Delta_SD_fit(kbar_t(ndt*dt)))
            if printSwitchD:
                print("Divisions: "+str(divisionNumber))
        else:
            #integrate forward by dt
            LNew, wNew = update_state(cellLengths[ndt], cellWidths[ndt], kbar_t(
                ndt*dt), kNew, lambdaNew, w0current, dt)
            cellLengths = np.append(cellLengths, LNew)
            cellWidths = np.append(cellWidths, wNew)
            ndt += 1
            #Check for stuck cell (very low probability but possible with distributions without a hard cut-off)
            if ndt*dt-dt*newCellTimestamps[divisionNumber] > 100:
                print("Cell is stuck")
                #Delete bad data
                for i in reversed(range(int(newCellTimestamps[int(len(newCellTimestamps))-1])+1,int(len(cellLengths)))):
                    cellLengths = np.delete(cellLengths,i)
                    cellWidths = np.delete(cellWidths,i)
                    ndt-=1
                #pick new length growth parameters
                while True:
                    junk1, junk2, junk3, kNew, lambdaNew = cell_division(cellLengths[ndt], cellWidths[ndt], w0current, kbar_t(ndt*dt))
                    if kNew>0 and lambdaNew>0:
                        break
                #Reset k, lambda
                cellMeank[len(cellMeank)-1] =  kNew
                cellMeanlambda[len(cellMeank)-1] = lambdaNew
                print("Cell unstuck")
    return cellLengths, cellWidths, np.arange(0,(ndt+1)*dt,dt)

LList,WList,tList=FULL_SIMULATION(5, 0.005, True)

##############################################################################
    # Make plots
##############################################################################
    
plt.rcParams.update({'font.size': 16})

lengthFig = plt.figure(1)
plt.plot(tList, LList, color="blue")
plt.ylabel('L (\u03BCm)')
plt.xlabel('t (min)')
plt.xlim([0,tList[-1]])
plt.ylim([np.min(LList)-0.05, np.max(LList)+0.05])
lengthFig.savefig('Length.png', dpi=300,bbox_inches='tight')

widthFig = plt.figure(2)
plt.plot(tList, WList, color="blue")
plt.ylabel('w (\u03BCm)')
plt.xlabel('t (min)')
plt.xlim([0,tList[-1]])
plt.ylim([np.min(WList)-0.05, np.max(WList)+0.05])
widthFig.savefig('Width.png', dpi=300,bbox_inches='tight')

plt.show()