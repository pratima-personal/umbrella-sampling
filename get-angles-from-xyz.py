import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-bias', type=str, help='bias value to analyse')
parser.add_argument('-log', action='store_true', help='plot log of probability')
parser.add_argument('-start', type=int, default=0, help='where to start calculating angles (after equilibriation)')
parser.add_argument('-show', action='store_true', help='whether to show plot or not')
parser.add_argument('-reshape', action='store_true', help='whether to spatially average theta_z values or not')
args = parser.parse_args()

# read in data with only ligand atom positions
data = np.genfromtxt('lig.' + args.bias, delimiter=' ')
size = len(data)
x0 = 0.0
y0 = 0.0
z0 = 0.0
x1 = 0.0
y1 = 0.0
z1 = 0.0
f = open('theta' + args.bias + '.txt','w')
xb0 = 0.0
yb0 = -54.7217
zb0 = 0.0
xb1 = 82.4293
yb1 = 54.7217
zb1 = 81.0040

frames = (len(data) / (240 * 18))
start = args.start
start = args.start * 240 * 18

for i in range (start,size,18):
  if (i + 17 < size):
    x0 = data[i,1]
    y0 = data[i,2]
    z0 = data[i,3]
    x1 = data[i+9,1]
    y1 = data[i+9,2]
    z1 = data[i+9,3]
    x_vec = x1 - x0
    y_vec = y1 - y0
    z_vec = (z1 - z0) / (zb1 - zb0)
    z_vec = (z_vec - round(z_vec)) * (zb1 - zb0)
    th = np.arctan(z_vec / y_vec)
    f.write('%4.5f\n' %(th))

f.close()

hist_data = np.genfromtxt('theta' + args.bias + '.txt', delimiter=' ')
if args.reshape:
    hist_data = np.mean(hist_data.reshape((-1, 240)), axis=1)

bins = np.linspace(-1.70, 1.70, 100)
hist, bins = np.histogram(hist_data, bins = bins, density = True)
bin_centres = bins[1:] * 0.5 + bins[:-1] * 0.5
plt.figure()
if args.log:
    bin_centres = bin_centres[hist != 0]
    hist = hist[hist != 0]
    if args.show:
        plt.plot(bin_centres, -np.log(hist))
        plt.show()
else:
    if args.show:
        plt.plot(180.0*bin_centres/np.pi, hist, color='blue', marker='o')
        plt.show()

