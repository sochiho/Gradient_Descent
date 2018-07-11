from __future__ import division

'''A plot illustrating the use of the gradient decent method to fit theoretical model
to laboratory data, for radioactive decay counts, by minimising the reduced Chi squared'''

import numpy
import matplotlib.pyplot as pyplot
import time
import matplotlib.cm
import pylab

dat = numpy.loadtxt('cp_1516_data6.csv', delimiter=',', skiprows=1)
TIME = dat[:,0] # in seconds
RATE = dat[:,1]
ERR  = dat[:,3]


def fn(TIME, n0, tau):
    # model function for the rate
    model = (n0/tau) * numpy.exp(-TIME/tau) 
    return model

def reduced_chi_squared(X):
    '''This function calculates chi - squared value'''
    
    n0, tau = X 
    model = fn(TIME, n0, tau)  
    chi_squared = ((model - RATE) / ERR) ** 2
    n_data = len(chi_squared)
    n_dof  = len(X) # degrees of freedom
    nu = n_data - n_dof
    return chi_squared.sum() / nu

def grad(X):
    '''This function returns the vector gradient of the function'''
    
    n0, tau = X
    d_n0 = 10e-6
    d_tau = 10e-6

    df_d_n0 = (reduced_chi_squared([n0 + d_n0/2, tau]) - reduced_chi_squared([n0 - d_n0/2, tau]))/d_n0

    df_d_tau = (reduced_chi_squared([n0, tau + d_tau/2]) - reduced_chi_squared([n0, tau - d_tau/2]))/d_tau

    return numpy.array((df_d_n0, df_d_tau))


def gradient_descent(X):
    '''This function for the gradient descent method'''
    
    n0, tau = X
    ga = 500 # Value of gamma
    it = 2500 # no of iterations

    chi_position = numpy.zeros((it, 2))
    chi_values = numpy.zeros((it))
    time_values = numpy.zeros((it))
    t0 = time.time()

    for i in range(it):
        x = X - ga * grad(X)
        chi_position[i] = x
        chi_values[i] = reduced_chi_squared(x)
        time_values[i] = time.time()
        X = x
        
    return chi_position, chi_values, time_values

# Parameters
n0_min, n0, n0_max = 10000, 11960, 14000
tau_min, tau, tau_max = 180, 230, 280
X0 = numpy.array((n0, tau))

N_points = 400
n0_axis = numpy.arange(n0_min, n0_max, (n0_max - n0_min)/N_points)
tau_axis = numpy.arange(tau_min,tau_max, (tau_max - tau_min)/N_points)

chi_position, chi_values, time_values = gradient_descent(X0)


#2D contour plot

DAT = numpy.zeros((len(tau_axis), len(n0_axis)))
for itau, tau in enumerate(tau_axis):
    for in0, n0 in enumerate(n0_axis):
        DAT[itau, in0] = reduced_chi_squared([n0, tau])


n0_position = chi_position[:,0]
tau_position = chi_position[:,1]

pyplot.figure(figsize = (12,6))
pyplot.subplot(211)
norm = matplotlib.colors.Normalize(vmin = 0, vmax = 12)
im = pyplot.imshow(DAT, extent = (n0_min, n0_max, tau_min, tau_max),
                   origin = 'lower', cmap = matplotlib.cm.gray,
                   norm = norm, aspect = 'auto')
pyplot.colorbar(im, orientation = 'vertical', label = 'Reduced Chi Squared')
pyplot.plot(n0_position, tau_position, color = 'white')
pyplot.xlabel('Initial number of atoms')
pyplot.ylabel('Tau /s')
pyplot.title('Reduced Chi Squared for values of n0 and tau.')
pyplot.tight_layout()

#a plot of the reduced chi squared values

pyplot.subplot(212)
pyplot.plot(time_values, chi_values)
pyplot.xlabel('Time /s')
pyplot.ylabel('Reduced Chi Squared')
pyplot.tight_layout()
pylab.title('Initial fit - Chi Squared = %.2f' % reduced_chi_squared((n0, tau)))

n0_best = n0_position[-1]
tau_best = tau_position[-1]
chi_best = reduced_chi_squared((n0_best,tau_best))
ANSWER1 = "The minimum reduced Chi-squared of %.2f occurs at N0 = %.2f and tau = %.2f"  % (chi_best, n0_best, tau_best)

print ANSWER1

pyplot.show()

