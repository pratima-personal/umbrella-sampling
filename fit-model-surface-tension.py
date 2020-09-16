import numpy as np
import argparse
import scipy.integrate as integrate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import rcParams

lwidth = 4.0
plt.rc('text', usetex=True, fontsize=28)
rcParams['text.latex.preamble'] = [r'\usepackage{helvet} \usepackage{sfmath}', r'\usepackage{upgreek}']
rcParams['axes.linewidth'] = 1.0*lwidth
rcParams['xtick.major.width'] = 1.0*lwidth
rcParams['xtick.major.size']  = 2.0*lwidth
rcParams['ytick.major.width'] = 1.0*lwidth
rcParams['ytick.major.size']  = 2.0*lwidth
plt.rc('lines', linewidth=4)
plt.rc('legend', frameon=False)

def gaussian(x, x0, v, a):
    return a * np.exp(-0.5 * (x - x0)**2 / v)

parser = argparse.ArgumentParser(description='')
parser.add_argument('-show_plots', action='store_true', help='whether to show diagnostic plots or not')
args = parser.parse_args()

# constants
kB = 1.3806503 * 6.0221415 / 4184.0
Lx = 8.1                            # simulation box length
N = 240                             # number of ligands

# find surface tension at transition temperature
data = np.genfromtxt('f_i-combined-375.6.txt')
theta_axis = data[:,0]
f_i = data[:,1]
prob_i = np.exp(-f_i)
trans_temp = 375.6
trans_beta = 1/(kB * trans_temp)

# fit to gaussians to get mean and variance
ord_spring = np.where((theta_axis >= -0.75) * (theta_axis <=-0.68))
x_ord = theta_axis[ord_spring]
y_ord = prob_i[ord_spring]
popt, pcov = curve_fit(gaussian, x_ord, y_ord, p0=( np.mean(y_ord), np.var(y_ord), max(y_ord) ))
mean_ord = popt[0]
v_ord = popt[1]
v_fit_ord = v_ord * N
norm_ord = popt[2]

disord_spring = np.where((theta_axis >= -0.50) * (theta_axis <=-0.11))
x_disord = theta_axis[disord_spring]
y_disord = prob_i[disord_spring]
popt, pcov = curve_fit(gaussian, x_disord, y_disord, p0=( np.mean(y_ord), np.var(y_ord), max(y_ord) ))
mean_disord = popt[0]
v_disord = popt[1]
v_fit_disord = v_disord * N
norm_disord = popt[2]

print(f'transition temp minima: ord = {mean_ord} disord = {mean_disord}')
print(f'transition temp sigma: ord = {v_ord} disord = {v_disord}')

p1 = gaussian(theta_axis, mean_ord, v_ord, norm_ord)
p2 = gaussian(theta_axis, mean_disord, v_disord, norm_disord)

if args.show_plots:
    # diagnostic plot #1
    plt.plot(theta_axis, p1, 'bv', label='ordered prob dist')
    plt.plot(theta_axis, p2, 'g^', label='disordered prob dist')
    plt.plot(theta_axis, prob_i, 'ro', label='total prob dist')
    plt.legend(loc='best')
    plt.show()

    # diagnostic plot #2
    plt.plot(theta_axis,-np.log(p1), 'bv', label=r'\boldsymbol{-} \textbf{log(} \boldsymbol{P_{ord}} \textbf{)}')
    plt.plot(theta_axis, -np.log(p2), 'g^', label=r'\boldsymbol{-} \textbf{log(} \boldsymbol{P_{disord}} \textbf{)}')
    plt.plot(theta_axis, -np.log(prob_i), 'ro')
    plt.xlabel(r'\boldsymbol{$\langle\theta_z\rangle}', fontsize=28)
    plt.ylabel(r'$\textbf{log probability}$', fontsize=28)
    plt.ylabel(r'$\boldsymbol{\beta F}$', fontsize=28)
    plt.legend(loc='best', markerscale=2)
    plt.show()

    # diagnostic plot #3
    g = prob_i - (p1 + p2)
    plt.plot(theta_axis, p1, 'bv', label='ordered prob dist')
    plt.plot(theta_axis, p2, 'g^', label='disordered prob dist')
    plt.plot(theta_axis, prob_i, 'ro', label='total prob dist')
    plt.plot(theta_axis, g, 'ks', label='difference in prob dist')
    plt.legend(loc='best')
    plt.show()

    # diagnostic plot #4
    plt.plot(theta_axis,-np.log(prob_i - (p1 + p2)), 'bv', label='log difference in prob dist')
    plt.plot(theta_axis, -np.log(prob_i), 'ro', label='log total prob dist')
    plt.legend(loc='best')
    plt.show()

# fitting functions to free energy profile for calculating surface tension
def surf_term(f, sigma):
    return np.exp( -trans_beta * sigma * Lx )

def integrand(f, args):
    x, sigma = args
    num = -N * ( -mean_ord * f + mean_disord * (f - 1) + x )**2 / ( 2 * (f * (v_fit_ord - v_fit_disord) + v_fit_disord) )
    denom = np.sqrt(2 * np.pi) * np.sqrt(v_fit_disord / N) * np.sqrt( v_fit_ord / (f*N - N * f**2) )
    denom = denom * np.sqrt( f * N * ( 1/v_fit_ord + f/( v_fit_disord * (1-f) ) ) )
    return (np.exp(num) * surf_term(f, sigma)) / denom

def curve(x, sigma):
    res = integrate.quad(integrand, 0.15, 0.9, [x,sigma])
    ord_term = gaussian(x, mean_ord, v_ord, norm_ord)
    disord_term = gaussian(x, mean_disord, v_disord, norm_disord)
    return -np.log( res[0] + ord_term + disord_term )

# fit to surface tension equation
vcurve = np.vectorize(curve)

# plot what free energy fit looks like for different values of sigma
size = np.logspace(np.log10(2.5), -0.5, 10).shape[0]
color = iter(plt.cm.rainbow(np.linspace(0,1,size)))
plt.figure(figsize=(14,9))
for sigma_i in np.logspace(np.log10(2.5), -0.5, 10):
    c = next(color)
    test_data = vcurve(theta_axis, sigma_i)
    plt.plot(theta_axis, test_data, 'o', color=c,
             label=r'$\mathsf{{fit\,\,for\,\,}}\sigma \mathsf{{\,\,=\,\,{:1.3f}\,\,kcal/mol}}'
                   r'\mbox{{-}}\mathsf{{nm}}$'.format(sigma_i/trans_beta))
    plt.plot(theta_axis, test_data, color=c, linewidth=4, alpha=0.5)
plt.xlabel(r'$\langle\theta_z\rangle')
plt.ylabel(r'$\beta \mathsf{F}(\langle\theta_z\rangle)$')
plt.legend(loc=(0.43,0.32), fontsize=22)
plt.ylim(-2,35)
plt.savefig('surf-tens-fit-range.pdf')

size = np.logspace(np.log10(2.5), -0.5, 10).shape[0]
color = iter(plt.cm.rainbow(np.linspace(0,1,size)))

# fit to get best surface tension value
popt, pcov = curve_fit(vcurve, theta_axis, f_i, p0=0.17)
print(f'results of surface tension optimisation (in kBT/nm): {popt[0]}')
print(f'results of surface tension optimisation (in kcal/mol-nm): {popt[0]/(trans_beta)}')

sigma_opt = popt[0]
test_data = vcurve(theta_axis, sigma_opt)
plt.figure(figsize=(14,9))
plt.plot(theta_axis, f_i, 'k', linewidth=4, label=r'$\mathsf{free\,\,energy\,\,density\,\,at\,\,}T_t$')
plt.plot(theta_axis, test_data, 'r', marker='v', markersize=10, linewidth=4, alpha=0.5,
         label=r'$\mathsf{{fit\,\,for\,\,}}\sigma ={:1.3f}\mathsf{{\,\,kcal/mol}}'
               r'\mbox{{-}}\mathsf{{nm}}$'.format(sigma_opt/trans_beta))
plt.xlabel(r'$\Theta_z\,\mathsf{(rad)}$')
plt.ylabel(r'$\mathsf{free\,\,energy\,\,density\,\,(kcal/mol}\mbox{-}\mathsf{rad)}$')
plt.xlim(-0.8, 0.0)
plt.legend(loc='upper right')
plt.savefig('best-surf-tens-coex-vac.pdf')