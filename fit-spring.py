import numpy as np
import argparse
import scipy.integrate as integrate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import rcParams

lwidth = 4.0
plt.rc('text', usetex=True, fontsize=28)
rcParams['text.latex.preamble'] = [r'\usepackage{helvet} \usepackage{sfmath}', r'\usepackage{upgreek}' ]
rcParams['axes.linewidth'] = 1.0*lwidth
rcParams['xtick.major.width'] = 1.0*lwidth
rcParams['xtick.major.size']  = 2.0*lwidth
rcParams['ytick.major.width'] = 1.0*lwidth
rcParams['ytick.major.size']  = 2.0*lwidth
plt.rc('lines', linewidth=4)
plt.rc('legend', frameon=False)

def parabola(x, x0, k, a, b):
    return a * 0.5 * k *(x - x0)**2 + b

parser = argparse.ArgumentParser(description='')
parser.add_argument('-temp', type=int, default=370, help='temperature')
args = parser.parse_args()

# constants
kB = 1.3806503 * 6.0221415 / 4184.0
N = 240                             # number of ligands

temp = 380.0
target_beta = 1 / (kB * args.temp)
data = np.genfromtxt('f_i-{}.txt'.format(args.temp))
theta_axis = data[:,0]
f_i = data[:,1]
prob_i = np.exp(-f_i)

ord_spring = np.where((theta_axis >= -0.74) * (theta_axis <=-0.68))
x_ord = theta_axis[ord_spring]
disord_spring = np.where((theta_axis >= -0.50) * (theta_axis <=-0.11))
x_disord = theta_axis[disord_spring]

# spring fit to free energy
y_ord = f_i[ord_spring]
popt, pcov = curve_fit( parabola, x_ord, y_ord, p0=(np.mean(x_ord), 1/np.var(y_ord), 10, 0 ))
mean_ord = popt[0]
k_ord = popt[1]
norm_ord = popt[2]
shift_ord = popt[3]
fit_ord = parabola(x_ord, mean_ord, k_ord, norm_ord, shift_ord)
print(popt[0], popt[1], popt[2])

y_disord = f_i[disord_spring]
popt, pcov = curve_fit( parabola, x_disord, y_disord, p0=(np.mean(x_disord), 1/np.var(y_disord), 10, 10 ))
mean_disord = popt[0]
k_disord = popt[1]
norm_disord = popt[2]
shift_disord = popt[3]
fit_disord = parabola(x_disord, mean_disord, k_disord, norm_disord, shift_disord)
print(popt[0], popt[1], popt[2])

# plot original free energy and fits
plt.plot(theta_axis, f_i, marker='o', color='#8856a7')
plt.plot(theta_axis, f_i, color='#8856a7', linewidth=4, alpha=0.5)
plt.plot(x_ord, fit_ord, 'bv')
plt.plot(x_disord, fit_disord, 'g^')
plt.xlabel(r'$\langle\theta_z\rangle$', fontsize=28)
plt.ylabel(r'$F(\langle\theta_z\rangle) \textrm{ (in kcal/mol-rad)}$', fontsize=28)
plt.show()

# calculate and print physical properties of phases
Q_ord = integrate.simps( np.exp(-fit_ord * target_beta), x_ord )
fullF_ord = - np.log(Q_ord) / target_beta
Q_disord = integrate.simps( np.exp(-fit_disord * target_beta), x_disord )
fullF_disord = - np.log(Q_disord) / target_beta

E_ord = integrate.simps(data[:,2][ord_spring], x_ord)
E_disord = integrate.simps(data[:,2][disord_spring], x_disord)

trans_guess = (E_disord - E_ord) / (S_disord - S_ord)
delS = (E_disord - E_ord) / args.temp + kB * np.log(Q_disord / Q_ord)

print(f'order parameter values: ord = {mean_ord} disord = {mean_disord}')
print(f'partition functions: ord = {Q_ord} disord = {Q_disord}')
print(f'total free energies: ord = {fullF_ord} disord = {fullF_disord}')
print(f'total energies: ord = {E_ord} disord = {E_disord}')
print(f'estimated transition by setting delta F = 0: {trans_guess} K')
print(f'differences: delta F = {fullF_disord-fullF_ord} delta E = {E_disord-E_ord} delta S = {delS}')



