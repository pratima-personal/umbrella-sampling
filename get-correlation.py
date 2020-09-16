import numpy as np
import argparse
import matplotlib.pyplot as plt
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

parser = argparse.ArgumentParser(description='')
parser.add_argument('-bias', type=str, help='bias value to analyse')
parser.add_argument('-steps', type=int, default=5000, help='number of time steps to analyse')
parser.add_argument('-pbc', action='store_true', help='whether to apply PBCs or not')
parser.add_argument('-savefigs', action='store_true', help='whether to save figures of CG lattice to make movie or not')
args = parser.parse_args()

data = np.genfromtxt('theta' + args.bias + '.txt', delimiter=' ')
data = data.reshape((-1,20,12))
Lx = 82.4293
Lz = 81.004

data_txz = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
data_txz[:, ::2, :] = data[:, 0:10, :]
data_txz[:, 1::2, :] = data[:, 10:20, :]
ligdata = data_txz
T = len(ligdata)

# get theta values at each lattice site as a function of time
theta_lat = []
for i in range(T):
    theta_lat.append( ligdata[i, :, :].flatten() )
theta_lat = np.array(theta_lat).reshape((-1, 20, 12))
theta_mean = np.mean(theta_lat, axis=0)

plt.figure(figsize=(12,9))
plt.imshow(theta_mean.T, aspect=1/0.6, cmap='PRGn_r', origin='lower', interpolation='none', vmin=-0.9, vmax=-0.1)
plt.yticks(np.arange(0, 12, 1))
plt.xticks(np.arange(0, 20, 1))
plt.ylim(-0.5,11.5)
plt.xlim(-0.5,19.5)
for i in np.arange(-0.5,12,1.0):
    plt.hlines(i, -0.5, 19.5, linestyle='solid')
for i in np.arange(-0.5,19,1.0):
    plt.vlines(i, -0.5, 11.5, linestyle='solid')
plt.xlabel('X', fontsize=28)
plt.ylabel('Z', fontsize=28)
plt.colorbar()
plt.savefig(args.bias+'-lat.pdf')

theta_lat -= theta_lat.mean()

if args.savefigs:
    name_arr = range(0, theta_lat.shape[0], 10)
    name_arr = np.array(name_arr)
    for j in range(len(name_arr)):
    # for j in range(0, 100, 20):
        matr = theta_lat[name_arr[j]].transpose()
        plt.imshow(matr, aspect=1.7, cmap='seismic_r', origin='lower', interpolation='none', vmin=-0.4, vmax=-0.1)
        plt.yticks(np.arange(0, 12, 1))
        plt.xticks(np.arange(0, 20, 1))
        plt.ylim(-0.5,11.5)
        plt.xlim(-0.5,19.5)
        for i in np.arange(-0.5,12,1.0):
            plt.hlines(i, -0.5, 19.5, linestyle='solid', linewidth=2)
        for i in np.arange(-0.5,19,1.0):
            plt.vlines(i, -0.5, 11.5, linestyle='solid', linewidth=2)
        plt.colorbar()
        plt.savefig('lat-' + args.bias + '-{:05d}.png'.format(j))
        plt.clf()

all_corr_xz = []
samplez = 10
X = theta_lat.shape[1]
Z = theta_lat.shape[2]
DT = theta_lat.shape[0] / samplez
var_list = []
for sample in xrange(samplez):
    To = DT*sample
    Tf = DT*(sample+1)
    sub_txz = theta_lat[To:Tf, :, :]
    
    cov_xz = np.zeros((X/2, Z/2))
    for t in xrange(DT):
       for x in xrange(X/2):
            for z in xrange(Z/2):
                cov_xz += sub_txz[t, x, z] * sub_txz[t, x : x + X/2, z : z + Z/2]
    
    cov_xz /= (DT * X/2 * Z/2 )
    var_list.append(cov_xz[0,0])
    corr_xz = cov_xz / cov_xz[0,0]
    
    x = range(X/2)
    z = range(Z/2)
    xv, zv = np.meshgrid(x, z)
    all_corr_xz.append(corr_xz)

all_corr_xz = np.array(all_corr_xz)
m_corr_xz = np.mean(all_corr_xz, axis=0)
d_corr_xz = np.std (all_corr_xz, axis=0) / np.sqrt(samplez)
 
# correlation plots with first data point removed
plt.figure(figsize=(14,9))
plt.plot(range(0, Z/2), m_corr_xz[0,:], c='#01665e', label=r'$\textrm{Z}$', lw=4, marker='o', markersize=9)
plt.fill_between(range(0, Z/2), m_corr_xz[0,:] - d_corr_xz[0,:], m_corr_xz[0,:] + d_corr_xz[0,:], color='#01665e', alpha=0.4)
plt.plot(range(0, X/2), m_corr_xz[:,0], c='#8c510a', label=r'$\textrm{X}$', lw=4, marker='o', markersize=9)
plt.fill_between(range(0, X/2), m_corr_xz[:,0] - d_corr_xz[:,0], m_corr_xz[:,0] + d_corr_xz[:,0], color='#8c510a', alpha=0.4)
plt.hlines(0, -0.1, X/2, linestyles='dashed')
plt.xlim(0.,X/2)
plt.ylim(-1.05,1.05)
plt.xlabel(r'$\mathsf{r}$')
plt.ylabel(r'$\mathsf{G}_\theta\mathsf{(r)}$')
plt.legend(loc='upper right')
plt.savefig('corr-thetaz-{}K.pdf'.format(args.bias))