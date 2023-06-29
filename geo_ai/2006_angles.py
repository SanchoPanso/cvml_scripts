import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


trajectory = pd.read_csv(r'C:\Users\HP\Downloads\Telegram Desktop\reference.csv', sep='\t')
trajectory = trajectory[
    [
        'projectedX[m]', 'projectedY[m]', 'projectedZ[m]',
        'heading[deg]',	'pitch[deg]', 'roll[deg]',   
    ]
]

X = trajectory['projectedX[m]'].to_numpy()
Y = trajectory['projectedY[m]'].to_numpy()
Z = trajectory['projectedZ[m]'].to_numpy()


H = trajectory['heading[deg]'].to_numpy()
P = trajectory['pitch[deg]'].to_numpy()
R = trajectory['roll[deg]'].to_numpy()

offsets = [3.69600e+05, 2.68183e+06, 2.40000e+02]
scales = [0.0001, 0.0001, 0.0001]

X = (X - offsets[0]) / scales[0]
Y = (Y - offsets[1]) / scales[1]
Z = (Z - offsets[2]) / scales[2]


installation_matrix = np.fromstring(
    "-1.00000000000000E+00 1.22464679914735E-16 1.49975978266186E-32 2.31849365000000E-01 "\
    "1.22464679914735E-16 1.00000000000000E+00 1.22464679914735E-16 0.00000000000000E+00 "\
    "0.00000000000000E+00 1.22464679914735E-16 -1.00000000000000E+00 -4.00589653000000E-01 "\
    "0.00000000000000E+00 0.00000000000000E+00 0.00000000000000E+00 1.00000000000000E+00", sep=' ').reshape(4, 4)

refinement_matrix = np.fromstring(
    "9.99992411163090E-01 1.74531175163006E-03 -3.48303073759075E-03 -1.39350660754115E-03 "\
    "-1.73008696422847E-03 9.99988957648521E-01 4.36936838924591E-03 2.15144333092834E-03 "\
    "3.49061818673809E-03 -4.36330928474669E-03 9.99984388436520E-01 -8.15551440850049E-04 "\
    "0.00000000000000E+00 0.00000000000000E+00 0.00000000000000E+00 1.00000000000000E+00", sep=' ', 
    dtype='float32').reshape(4, 4)

installation_matrix[0, 3] /= scales[0]
installation_matrix[1, 3] /= scales[1]
installation_matrix[2, 3] /= scales[2]

refinement_matrix[0, 3] /= scales[0]
refinement_matrix[1, 3] /= scales[1]
refinement_matrix[2, 3] /= scales[2]


start = 434
stop = 434 + 1 #644

xs = X[start:stop]
ys = Y[start:stop]
zs = Z[start:stop]

zs[:] = 0


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xs, ys, zs, 'o', color='#00ff00')

for i in range(start, stop):
    rays = np.array(
        [
            [1., 0., 0., 1.],
            [0., 1., 0., 1.],
            [0., 0., 1., 1.],
        ]
    )
    
    rays *= 50000
    
    h_rad = (H[i] - 90) * np.pi / 180
    p_rad = P[i] * np.pi / 180
    r_rad = R[i] * np.pi / 180
    
    roll_rot = np.array(
        [
            [1,             0,              0, 0],
            [0, np.cos(r_rad), -np.sin(r_rad), 0],
            [0, np.sin(r_rad),  np.cos(r_rad), 0],
            [0,             0,              0, 1],
        ]
    )
    pitch_rot = np.array(
        [
            [ np.cos(p_rad), 0, np.sin(p_rad), 0],
            [             0, 1,             0, 0],
            [-np.sin(p_rad), 0, np.cos(p_rad), 0],
            [0,             0,              0, 1],
        ]
    )
    heading_rot = np.array(
        [
            [np.cos(h_rad), -np.sin(h_rad), 0, 0],
            [np.sin(h_rad),  np.cos(h_rad), 0, 0],
            [            0,              0, 1, 0],
            [0,             0,              0, 1],
        ]
    )
    xz_rot = np.array(
        [
            [ 0., 0., -1., 0.],
            [ 0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ]
    )
    rays = rays @ xz_rot @ roll_rot @ pitch_rot @ heading_rot
    #rays = (heading_rot @ pitch_rot @ roll_rot @ xz_rot @ rays.T)
    
    colors = ['#ff0000', '#00ff00', '#0000ff']
    for j, ray in enumerate(rays):
        xs = [0 + X[i], ray[0] + X[i]]
        ys = [0 + Y[i], ray[1] + Y[i]]
        zs = [0 + Z[i], ray[2] + Z[i]]
        
        ax.plot(xs, ys, zs, color=colors[j])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#plt.show()



for i in range(start, stop):
    rays_pt1 = np.array(
        [
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
        ]
    )
    rays_pt2 = np.array(
        [
            [1., 0., 0., 1.],
            [0., 1., 0., 1.],
            [0., 0., 1., 1.],
        ]
    )
    
    rays_pt1 *= 50000
    rays_pt2 *= 50000
    
    h_rad = (H[i] - 90) * np.pi / 180
    p_rad = P[i] * np.pi / 180
    r_rad = R[i] * np.pi / 180
    
    roll_rot = np.array(
        [
            [1,             0,              0, 0],
            [0, np.cos(r_rad), -np.sin(r_rad), 0],
            [0, np.sin(r_rad),  np.cos(r_rad), 0],
            [0,             0,              0, 1],
        ]
    )
    pitch_rot = np.array(
        [
            [ np.cos(p_rad), 0, np.sin(p_rad), 0],
            [             0, 1,             0, 0],
            [-np.sin(p_rad), 0, np.cos(p_rad), 0],
            [0,             0,              0, 1],
        ]
    )
    heading_rot = np.array(
        [
            [np.cos(h_rad), -np.sin(h_rad), 0, 0],
            [np.sin(h_rad),  np.cos(h_rad), 0, 0],
            [            0,              0, 1, 0],
            [0,             0,              0, 1],
        ]
    )
    xz_rot = np.array(
        [
            [ 0., 0., -1., 0.],
            [ 0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ]
    )
    rays_pt1 = rays_pt1 @ installation_matrix @ xz_rot @ roll_rot @ pitch_rot @ heading_rot
    rays_pt2 = rays_pt2 @ installation_matrix @ xz_rot @ roll_rot @ pitch_rot @ heading_rot
    
    colors = ['#ff5555', '#55ff55', '#5555ff']
    for j in range(len(rays_pt1)):
        xs = [rays_pt1[j][0] + X[i], rays_pt2[j][0] + X[i]]
        ys = [rays_pt1[j][1] + Y[i], rays_pt2[j][1] + Y[i]]
        zs = [rays_pt1[j][2] + Z[i], rays_pt2[j][2] + Z[i]]
        
        ax.plot(xs, ys, zs, color=colors[j])

plt.show()


