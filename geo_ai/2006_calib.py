import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


trajectory = pd.read_csv(r'C:\Users\HP\Downloads\Telegram Desktop\reference.csv', sep='\t')

X = trajectory['projectedX[m]'].to_numpy()
Y = trajectory['projectedY[m]'].to_numpy()
Z = trajectory['projectedZ[m]'].to_numpy()

H = trajectory['heading[deg]'].to_numpy()
P = trajectory['pitch[deg]'].to_numpy()
R = trajectory['roll[deg]'].to_numpy()

idx = 433 + 25

pov = [X[idx], Y[idx], Z[idx]]


ray = np.array([[1., 0., 0.]]) * 5
h_rad = -(H[idx] - 90) * np.pi / 180
p_rad = -(P[idx]) * np.pi / 180 # (P[idx] - 90)
r_rad = -(R[idx]) * np.pi / 180

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
trans = np.array(
    [
        [1, 0, 0, pov[0]],
        [0, 1, 0, pov[1]],
        [0, 0, 1, pov[2]],
        [0, 0, 0,      1],
    ]
)
xy_rot = np.array(
    [
        [0., -1., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ]
)

preinst_oy_rot = np.array(
    [
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [-1., 0., 0., 0.],
        [0., 0., 0., 1.],
    ]
)

preinst_oz_rot = np.array(
    [
        [0., 1., 0., 0.],
        [-1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ]
)

pseudo_inst = np.array(
    [
        [-1., 0., 0., -30000.],
        [0., 1., 0., 10000.],
        [0., 0., -1., 0.],
        [0., 0., 0., 1.],
    ]
)

extrinsic_inv = trans @ heading_rot @ pitch_rot @ roll_rot
extrinsic = np.linalg.inv(extrinsic_inv)


object_points = np.array(
    [   
        # # 25_2
        # [369520.940300, 2681851.457701, 237.075394],
        # [369527.002998, 2681850.810200, 237.340195],
        # [369533.004402, 2681850.936701, 237.183701],
        # [369534.051102, 2681849.673901, 237.164902],
        # [369547.639500, 2681849.093000, 237.143997],
        # [369549.373798, 2681849.919201, 237.380707],
        # [369553.558399, 2681841.628500, 237.315399],
        # [369540.664700, 2681842.211000, 237.359497],
        # [369528.866997, 2681841.867200, 237.438202],
        
        # 25_0
        [369501.753799, 2681844.320100, 236.940399],
        [369500.110001, 2681844.479200, 236.935898],
        [369496.218002, 2681844.671200, 236.872498],
        # [369494.544601, 2681844.911800, 236.708801],
        # [369493.017502, 2681848.873400, 236.594696],
        # [369496.453796, 2681851.708500, 236.730804],
        [369500.344002, 2681851.440399, 236.811295],
    ],
dtype='float32') 

# Points in 2d image
image_points = np.array(
    [   
        # # 25_2
        # [220, 1613],
        # [670, 1272],
        # [780, 1184],
        # [865, 1189],
        # [924, 1120],
        # [890, 1118],
        # [1130, 1108],
        # [1204, 1144],
        # [1455, 1223],
        
        # 25_0
        [639, 1211],
        [700, 1177],
        [752, 1149],
        # [755, 1145],
        # [965, 1137],
        # [1121, 1147],
        [1144, 1177],
    ],  
dtype='float32')

# Intrisic camera matrix
fx = 1024
fy = 1024
cx = 1024
cy = 1024
camera_matrix = np.array(
    [
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ],
dtype='float32')

retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_SQPNP)

def get_homography(rvec, tvec):
    h_mat = np.eye(4, dtype='float32')
    rot_mat = cv2.Rodrigues(rvec)[0]
    h_mat[0:3, 0:3] = rot_mat
    h_mat[0:3, 3:4] = tvec
    
    return h_mat

actual_extrinsic = get_homography(rvec, tvec)
inst = actual_extrinsic @ np.linalg.inv(extrinsic)
print(inst)
print(np.linalg.inv(inst)[0:3, 3])

# rvec = cv2.Rodrigues(extrinsic[0:3, 0:3])[0]
# tvec = extrinsic[0:3, 3:4]


img = np.zeros((2048, 2048, 3), dtype='uint8')
res_image_points = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)[0].reshape(-1, 2)

for pt in image_points:
    cv2.circle(img, (int(pt[0]), int(pt[1])), 20, (0, 255, 0), -1)
    
for pt in res_image_points:
    cv2.circle(img, (int(pt[0]), int(pt[1])), 15, (0, 0, 255), -1)
    
plt.imshow(img[:, :, ::-1])
plt.show()


# [[-1.39528896e-01 -9.90198006e-01 -6.28864569e-03  2.98430799e+03]
#  [ 2.63899165e-02  2.63005891e-03 -9.99648285e-01  1.47012168e+03]
#  [ 9.89866301e-01 -1.39645779e-01  2.57642746e-02 -2.96520837e+04]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]


# inst 25_0
# [[ 0.0152225,   0.9987762,  -0.04705681,  0.17976795],
#  [ 0.01645279, -0.04730609, -0.99874496,  0.26238015],
#  [-0.99974879,  0.01442918, -0.01715277, -0.07896413],
#  [ 0.,          0.,          0.,          1.        ]]


# pov 25 inst [ 0.      -0.40058965  0.23184937]

# pov 25_0 [-0.0859977  -0.16599639  0.26915569]
# pov 25_2 [-0.1676281  -0.26721376  0.23555786]