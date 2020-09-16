import numpy as np
import scipy.special as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import argparse
from matplotlib import rcParams

lwidth = 4.0
plt.rc('text', usetex=True, fontsize=30)
rcParams['text.latex.preamble'] = [r'\usepackage{helvet} \usepackage{sfmath}', r'\usepackage{upgreek}']
rcParams['axes.linewidth'] = 1.0*lwidth
rcParams['xtick.major.width'] = 1.0*lwidth
rcParams['xtick.major.size']  = 2.0*lwidth
rcParams['ytick.major.width'] = 1.0*lwidth
rcParams['ytick.major.size']  = 2.0*lwidth
plt.rc('lines', linewidth=4)
plt.rc('legend', frameon=False)

def bessel_0(x):
    return sp.kn(0, x)

def chi_func(F, L, x, y, dy):
    assert( len(x) == len(y) )
    chisq = 0
    for i in xrange(len(x)):
        err   = y[i] - F(x[i] / float(L) )
        err_s = err / dy[i]
        chisq += err_s * err_s
    chisq /= len(x)
    return chisq

parser = argparse.ArgumentParser()
parser.add_argument('-temp', type=str, help='Temperature at which to get correlation')
parser.add_argument('-v', action='store_true', help='Run verbosely')
parser.add_argument('-use_x', action='store_true', help='Use x-correlation for fitting, in false, use z-correlation')
args = parser.parse_args()

sheet_data = np.genfromtxt(f'cg-corr-{args.temp}K.txt')
x_max = int(max(sheet_data[:,0]))
z_max = int(max(sheet_data[:,1]))
corr_xz = np.zeros((x_max+1, z_max+1))
dcorr_xz = np.zeros((x_max+1, z_max+1))
for row in sheet_data:
    corr_xz[row[0], row[1]] = row[2]
    dcorr_xz[row[0], row[1]] = row[3]

# choose either x- or z-direction for fitting
if args.use_x:
    print('using x-correlation for fitting')
    axis = range(1,int(x_max+1))
    dat = corr_xz[1:, 0]
    d_dat = dcorr_xz[1:, 0]
else:
    print('using z-correlation for fitting')
    axis = range(1,int(z_max+1))
    dat = corr_xz[0, 1:]
    d_dat = dcorr_xz[0, 1:]

def chi_bessel(L):
    return chi_func(bessel_0, L, axis, dat, d_dat)
optL = opt.minimize(chi_bessel, .1).x[0]

def chi_shift(L):
    return abs(chi_bessel(L) - chi_bessel(optL) - 1)

varL = opt.minimize(chi_shift, optL*1.1, method='TNC', bounds=((optL,None),) ).x[0]
errL = varL - optL

if args.v:
    print('Minimum value of L +/- dL for K_0(x/L) fit to data: {optL} +/- {errL}')

# plot fit versus actual data
if args.use_x:
    x_fit = np.linspace(0.9, x_max, 25)
else:
    x_fit = np.linspace(0.9, z_max, 25)
y_fit = sp.kn(0, x_fit/optL)
y_maxfit = sp.kn(0, x_fit/(optL+2*errL))
y_minfit = sp.kn(0, x_fit/(optL-2*errL))

plt.figure(figsize=(12,9))
plt.plot(x_fit, y_fit, color='#a50f15', lw=3, label=r'$\mathsf{Bessel\,\,fit}$')
plt.fill_between(x_fit, y_minfit, y_maxfit, color='#a50f15', alpha=0.5)
plt.plot(axis, dat, color='#01665e', marker='v', markersize=22, lw=0, label=r'$\mathsf{Original\,\,data\,\,from\,\,simulation}$')
plt.xlim((0.8, max(x_fit) + 0.2))
plt.legend(loc='best')
plt.xlabel(r'$\mathsf{z}$')
plt.ylabel(r'$\mathsf{G(z)}$')
plt.savefig('cg-besselfit-z-{}K.pdf'.format(args.temp))


