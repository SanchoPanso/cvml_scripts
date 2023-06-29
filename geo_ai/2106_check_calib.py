import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





def main():
    project_different_images()

def get_rot_ox(angle: float) -> np.ndarray:
    """Get homography matrix that represents rotation aroung Ox

    :param angle: angle in radians
    :return: homography matrix
    """
    
    mat = np.array(
        [
            [1,             0,              0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle),  np.cos(angle), 0],
            [0,             0,              0, 1],
        ]
    )
    
    return mat


def get_rot_oy(angle: float) -> np.ndarray:
    """Get homography matrix that represents rotation aroung Oy

    :param angle: angle in radians
    :return: homography matrix
    """
    
    mat = np.array(
        [
            [ np.cos(angle), 0, np.sin(angle), 0],
            [             0, 1,             0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0,             0,              0, 1],
        ]
    )
    
    return mat


def get_rot_oz(angle: float) -> np.ndarray:
    """Get homography matrix that represents rotation aroung Oz

    :param angle: angle in radians
    :return: homography matrix
    """
    
    mat = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle),  np.cos(angle), 0, 0],
            [            0,              0, 1, 0],
            [            0,              0, 0, 1],
        ]
    )
    
    return mat

def get_translation(x:float, y:float, z:float)-> np.ndarray:
    """Get homography matrix that represents translation

    """
    
    mat = np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    ) 
    
    return mat


def project_different_images():
    
    installation_matrix = np.fromstring(
    "-1.00000000000000E+000 0.00000000000000E+000 0.00000000000000E+000 2.31849365000000E-001 0.00000000000000E+000 1.00000000000000E+000 0.00000000000000E+000 0.00000000000000E+000 0.00000000000000E+000 0.00000000000000E+000 -1.00000000000000E+000 -4.00589653000000E-001 0.00000000000000E+000 0.00000000000000E+000 0.00000000000000E+000 1.00000000000000E+000", sep=' ').reshape(4, 4)
    refinement_matrix = np.fromstring(
    "9.99992411163090E-01 1.74531175163006E-03 -3.48303073759075E-03 -1.39350660754115E-03 "\
    "-1.73008696422847E-03 9.99988957648521E-01 4.36936838924591E-03 2.15144333092834E-03 "\
    "3.49061818673809E-03 -4.36330928474669E-03 9.99984388436520E-01 -8.15551440850049E-04 "\
    "0.00000000000000E+00 0.00000000000000E+00 0.00000000000000E+00 1.00000000000000E+00", sep=' ').reshape(4, 4)
    object_points = np.array(
        [
            [369520.940300, 2681851.457701, 237.075394],
            [369527.002998, 2681850.810200, 237.340195],
            [369533.004402, 2681850.936701, 237.183701],
            [369534.051102, 2681849.673901, 237.164902],
            [369547.639500, 2681849.093000, 237.143997],
            [369549.373798, 2681849.919201, 237.380707],
            [369553.558399, 2681841.628500, 237.315399],
            [369540.664700, 2681842.211000, 237.359497],
            [369528.866997, 2681841.867200, 237.438202],
            
            [369493.057701, 2681848.708799, 236.598297],
            [369486.934196, 2681853.681499, 236.526299],
            [369486.759003, 2681875.764198, 236.731293],
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
    
    trajectory = pd.read_csv(r'C:\Users\HP\Downloads\Telegram Desktop\reference.csv', sep='\t')


    X = trajectory['projectedX[m]'].to_numpy()
    Y = trajectory['projectedY[m]'].to_numpy()
    Z = trajectory['projectedZ[m]'].to_numpy()


    H = trajectory['heading[deg]'].to_numpy()
    P = trajectory['pitch[deg]'].to_numpy()
    R = trajectory['roll[deg]'].to_numpy()
    
    start = 433 + 22
    stop = 433 + 46 #644
    
    
    img_paths = [
        r'D:\geo_ai_data\shots\5\pano_000005_000022_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000023_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000024_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000025_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000026_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000027_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000028_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000029_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000030_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000031_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000032_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000033_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000034_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000035_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000036_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000037_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000038_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000039_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000040_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000041_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000042_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000043_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000044_2.jpg',
        r'D:\geo_ai_data\shots\5\pano_000005_000045_2.jpg',
    ]
    
    
    for i in range(start, stop):
        
        print(X[i], Y[i], Z[i], (H[i] - 90), P[i], R[i])
        h_rad = -(H[i] - 90) * np.pi / 180
        p_rad = -P[i] * np.pi / 180
        r_rad = -R[i] * np.pi / 180

        
        roll_rot = get_rot_ox(r_rad)
        pitch_rot = get_rot_oy(p_rad)
        heading_rot = get_rot_oz(h_rad)
        
        trans = get_translation(X[i], Y[i], Z[i])
        
        y_rot1 = np.array(
            [
                [ 0, 0, 1, 0],
                [ 0, 1, 0, 0],
                [-1, 0, 0, 0],
                [ 0, 0, 0, 1],
            ]
        )
        x_rot2 = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        
        y_rot_180 = np.array(
            [
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )
        
        special_rot = np.array(
            [
                [-1, 0, 0, 2 * 0.13],
                [0, 1, 0, 0],
                [0, 0, -1, 2 * 0.16],
                [0, 0, 0, 1],
            ]
        )
        inst = np.array(
            [[-0.01017697, -0.99938181,  0.03365118, -0.27668132],
            [-0.01270778, -0.03352094, -0.99935723,  0.22431901],
            [ 0.99986744, -0.01059806, -0.01235878,  0.16768514],
            [ 0.,          0.,          0.,          1.,        ]]
        )
        # print(np.sqrt(0.27668132**2 + 0.22431901**2 + 0.16768514**2))
        # print(np.sqrt(2.31849365000000E-01**2 + 0.00000000000000E+00**2 + 4.00589653000000E-01**2))
        
        extrinsic_inv = trans @ heading_rot @ pitch_rot @ roll_rot
        
        #extrinsic = inst @ np.linalg.inv(extrinsic_inv)
        extrinsic = x_rot2 @ y_rot1 @ installation_matrix @ x_rot2 @ y_rot1 @ np.linalg.inv(extrinsic_inv)
        
        print(np.linalg.inv(x_rot2 @ y_rot1 @ installation_matrix @ x_rot2 @ y_rot1)[:3, 3])
        
        rvec = cv2.Rodrigues(extrinsic[0:3, 0:3])[0]
        tvec = extrinsic[0:3, 3:4]
        
        img = cv2.imread(img_paths[i - start])
        
        mask = (extrinsic @ np.concatenate([object_points, np.ones((len(object_points), 1))], axis=1).T)[2, :] >= 0

        points_2d = cv2.projectPoints(object_points[mask], rvec, tvec, camera_matrix, None)[0].reshape(-1, 2)
        
        for pt in points_2d:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 15, (255, 0, 0), -1)
        cv2.imshow('img', cv2.resize(img, (600, 600)))
        cv2.waitKey()
        cv2.imwrite(f'{i - start}.jpg', img)


if __name__ == '__main__':
    main()
