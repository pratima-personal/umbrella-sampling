import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-f', type=str, help='file to analyse')
parser.add_argument('-div', type=float, help='angle value which divides the phases')
args = parser.parse_args()

zb0 = 0.0
zb1 = 81.0040
n_atoms = 10320
mol_dat = []
Rnum_O = 240 * 18
Rnum_Ne = 240 * 18
O_used = 0
Ne_used = 0
data = []

# process a single frame and assign Ne or O to ligands
def analyseFrame(data):
    newdata = np.zeros((n_atoms, 5))
    all_O = False
    O_used = 0
    Ne_used = 0
    all_Ne = False
    in_mol=False
    count = 0

    for j in xrange(len(newdata)):
        if (data[j][0] == '2' and float(data[j][2]) > 0 and not in_mol):
            in_mol=True
            mol_dat = []
            mol_dat.append(data[j])
        elif (data[j][0] == '2' and float(data[j][2]) > 0 and in_mol):
            mol_dat.append(data[j])
        elif (data[j][0] == '1' and float(data[j][2]) > 0):
            if not in_mol:
                raise ValueError('Hit atom type 1 but not in a molecule!')
            mol_dat.append(data[j])
            in_mol=False
            y0 = float(mol_dat[0][2])
            z0 = float(mol_dat[0][3])
            y1 = float(mol_dat[9][2])
            z1 = float(mol_dat[9][3])
            y_vec = y1 - y0
            z_vec = (z1 - z0) / (zb1 - zb0)
            z_vec = (z_vec - round(z_vec)) * (zb1 - zb0)
            th = -np.arctan(z_vec / y_vec)
            if (abs(th) >= args.div):
                for k in xrange(j-17, j+1, 1):
                    newdata[count][0] = '8'
                    newdata[count][1] = data[k][1]
                    newdata[count][2] = data[k][2]
                    newdata[count][3] = data[k][3]
                    newdata[count][4] = th
                    O_used = O_used + 1
                    count = count + 1
            else:
               for k in xrange(j-17, j+1, 1):
                    newdata[count][0] = '10'
                    newdata[count][1] = data[k][1]
                    newdata[count][2] = data[k][2]
                    newdata[count][3] = data[k][3]
                    newdata[count][4] = th
                    Ne_used = Ne_used + 1
                    count = count + 1
        else:
           newdata[count][0] = data[j][0]
           newdata[count][1] = data[j][1]
           newdata[count][2] = data[j][2]
           newdata[count][3] = data[j][3]
           newdata[count][4] = -100
           count = count + 1

    # find number of remaining bath atoms
    O_left = Rnum_O - O_used
    Ne_left = Rnum_Ne - Ne_used
    return O_left, Ne_left, newdata

# write out atoms for that frame
def writeNewFrame(newdata, O_left, Ne_left, t):
    print('{n_atoms + Rnum_Ne} \n{t/n_atoms} Comment line')

    for j in xrange(n_atoms):
        # start by adding all the oxygens
        if (newdata[j][0] == 8.):
            print('{int(newdata[j][0])} {newdata[j][1]} {newdata[j][2]} {newdata[j][3]} {newdata[j][4]}')
    for k in xrange(O_left):
        print('8 -150 -150 -150 -100')

    for j in xrange(n_atoms):
        # now add all the bath heliums
        if (newdata[j][0] == 10.):
            print('{int(newdata[j][0])} {newdata[j][1]} {newdata[j][2]} {newdata[j][3]} {newdata[j][4]}')
    for k in xrange(Ne_left):
        print('10 -150 -150 -150 -100')

    for j in xrange(n_atoms):
        # now the CH3lig atoms
        if (newdata[j][0] == 2.):
            print('{int(newdata[j][0])} {newdata[j][1]} {newdata[j][2]} {newdata[j][3]} {newdata[j][4]}')

    for j in xrange(n_atoms):
        # now the CH2lig atoms
        if (newdata[j][0] == 1.):
            print('{int(newdata[j][0])} {newdata[j][1]} {newdata[j][2]} {newdata[j][3]} {newdata[j][4]}')

    for j in xrange(n_atoms):
        # now the remaining atoms
        if (newdata[j][0] == 3. or newdata[j][0] == 4. or newdata[j][0] == 5. or newdata[j][0] == 6.):
            print('{int(newdata[j][0])} {newdata[j][1]} {newdata[j][2]} {newdata[j][3]} {newdata[j][4]}')

# main run here
with open(args.f) as f:
    for t,l in enumerate(f):
        l_arr = l.split()
        data.append(l_arr)
        if (t%n_atoms == 0 and t > 0):
            data = np.array(data)
            O_left, Ne_left, newdata = analyseFrame(data)
            writeNewFrame(newdata, O_left, Ne_left, t)
            data = []

