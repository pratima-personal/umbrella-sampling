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

# make coarse grained lattice
cd_data = np.genfromtxt('paired-sites.txt')
n_cd = len(cd_data)

# blocks of two (average only in x-direction)
cg_lat = []
for i in range(n_cd):
    cg_lat.append((cd_data[i][2] + 4.12 * 0.5, cd_data[i][3], cd_data[i][4]))
cg_lat = np.array(cg_lat)
nn_dist = 4.12 * 0.5

cg_xz = np.transpose(np.array((cg_lat[:, 0], cg_lat[:, 2])))
full_indices = []
indices = []
# now loop over coarse grained lattice and get nearest neighbours for each point
for j in range(n_cd):
    x = cg_xz[j][0]
    z = cg_xz[j][1]
    # now find the Cd atoms closest to this coarse grained site
    dx = x - cd_data[:, 2]
    dz = z - cd_data[:, 4]
    if args.pbc:
        # apply PBCs
        dx = dx - Lx * np.round(dx / Lx)
        dz = dz - Lz * np.round(dz / Lz)
    dist = np.sqrt(dx ** 2 + dz ** 2)
    full_indices.append(np.where(dist <= nn_dist + 0.05))

for j in range(n_cd):
    indices.append(full_indices[j])

# now assign theta values to each coarse grained site as a function of time
cg_theta = []
t_steps = theta_lat.shape[0]
for i in range(T - t_steps, T):
    theta_t = []
    for j in range(len(indices)):
        theta_site = theta_lat[i - T + t_steps][indices[j]]
        theta_t.append(np.mean(theta_site))
    cg_theta.append(np.array(theta_t))

cg_theta = np.array(cg_theta)
cg_theta = cg_theta.reshape((-1, 20, 12))
cg_mean = np.mean(cg_theta, axis=0)

if args.savefigs:
    name_arr = range(0, t_steps, 10)
    name_arr = np.array(name_arr)
    for j in range(len(name_arr)):
        matr = cg_theta[name_arr[j]].transpose()
        plt.imshow(matr, aspect=1.7, cmap='PRGn_r', origin='lower', interpolation='none', vmin=-0.8, vmax=-0.1)
        plt.yticks(np.arange(0, 12, 1))
        plt.xticks(np.arange(0, 20, 1))
        plt.ylim(-0.5, 11.5)
        plt.xlim(-0.5, 19.5)
        for i in np.arange(-0.5, 12, 1.0):
            plt.hlines(i, -0.5, 19.5, linestyle='solid', linewidth=2)
        for i in np.arange(-0.5, 19, 1.0):
            plt.vlines(i, -0.5, 11.5, linestyle='solid', linewidth=2)
        plt.colorbar()
        plt.savefig('cg-lat-' + args.bias + '-{:05d}.png'.format(j))
        plt.clf()

th_av = np.mean(cg_theta)
th_std = np.std(cg_theta)
beta = 1 / (400 * 1.3806503 * 6.0221415 / 4184.0)
bins = np.linspace(-1.0, 0.5, 100)
plt.hist(cg_theta[:, :, :].flatten(), bins=bins, normed=True, histtype='stepfilled', alpha=0.7, color='green')
plt.xlabel(r'$\boldsymbol{\phi}$', fontsize=28)
plt.ylabel(r'$\boldsymbol{P(\phi)}$', fontsize=28)
plt.title(r'$\textbf{Probability distribution of 2-site average angle}$', fontsize=28)
plt.show()

plt.figure(figsize=(12, 9))
plt.imshow(cg_mean.T, aspect=1 / 0.6, cmap='PRGn_r', origin='lower', interpolation='none', vmin=-0.8, vmax=-0.1)
plt.yticks(np.arange(0, 12, 1))
plt.xticks(np.arange(0, 20, 1))
plt.ylim(-0.5, 11.5)
plt.xlim(-0.5, 19.5)
for i in np.arange(-0.5, 12, 1.0):
    plt.hlines(i, -0.5, 19.5, linestyle='solid')
for i in np.arange(-0.5, 19, 1.0):
    plt.vlines(i, -0.5, 11.5, linestyle='solid')
plt.xlabel('Z', fontsize=28)
plt.ylabel('X', fontsize=28)
plt.colorbar()
plt.savefig(args.bias + '-cg-lat.pdf')

X = cg_mean.shape[0]
Z = cg_mean.shape[1]

mean = np.mean(cg_theta[:, :, :], axis=(0, 1, 2))
cg_theta[:, :, :] -= mean

all_corr_xz = []
samplez = 20
DT = cg_theta.shape[0] / samplez
dist_xz = np.zeros((X / 2, Z / 2))
for sample in xrange(samplez):
    To = DT * sample
    Tf = DT * (sample + 1)
    sub_txz = cg_theta[To:Tf, :, :]

    cov_xz = np.zeros((X / 2, Z / 2))
    for t in xrange(DT):
        for x in xrange(X / 2):
            for z in xrange(Z / 2):
                cov_xz += sub_txz[t, x, z] * sub_txz[t, x: x + X / 2, z: z + Z / 2]
                dist_xz[x, z] = np.sqrt((4.12 * x) ** 2 + (6.75 * z) ** 2)

    cov_xz /= (DT * X / 2 * Z / 2)
    corr_xz = cov_xz / cov_xz[0, 0]

    x = range(X / 2)
    z = range(Z / 2)
    xv, zv = np.meshgrid(x, z)
    all_corr_xz.append(corr_xz)

all_corr_xz = np.array(all_corr_xz)
m_corr_xz = np.mean(all_corr_xz, axis=0)
d_corr_xz = np.std(all_corr_xz, axis=0) / np.sqrt(samplez)

# correlation plot in both directions
plt.figure(figsize=(14, 9))
plt.plot(range(0, Z/2), m_corr_xz[0, :], c='#01665e', label=r'$\textrm{Z}$', lw=4, marker='o', markersize=9)
plt.fill_between(range(0, Z/2), m_corr_xz[0, :]-d_corr_xz[0, :], m_corr_xz[0, :]+d_corr_xz[0, :], color='#01665e',
                 alpha=0.4)
plt.plot(range(0, X/2), m_corr_xz[:, 0], c='#8c510a', label=r'$\textrm{X}$', lw=4, marker='o', markersize=9)
plt.fill_between(range(0, X/2), m_corr_xz[:, 0]-d_corr_xz[:, 0], m_corr_xz[:, 0]+d_corr_xz[:, 0], color='#8c510a',
                 alpha=0.4)
plt.hlines(0, -0.1, X / 2, linestyles='dashed')
plt.xlim(-0.1, X / 2)
plt.ylim(-0.2, 1.05)
plt.xlabel(r'$\mathsf{r}$')
plt.ylabel(r'$\mathsf{G}_\phi\mathsf{(r)}$')
plt.legend(loc='upper right')
plt.savefig('cg-corr-xz-{}K.pdf'.format(args.bias))
plt.clf()

# print out correlations to file
for i in range(X / 2):
    for j in range(Z / 2):
        print(f'{i} {j} {m_corr_xz[i, j]} {d_corr_xz[i, j]}')
