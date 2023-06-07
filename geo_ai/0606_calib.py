import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


points_flat_4 = [
    [1192, 602],
    [122, 635],
    [746, 539],
    [596, 545],
    [524, 469],
    [700, 460],
]

# dif = [[-75087.125 216060.1   -54946.094]]

def main():
    h, rvec, tvec = solve_pnp_pov()
    # h = solve_pnp()
    
    # show_2d_point()
    #project_3d_to_flat(h)
    # find_dif_h(h)
    # apply_h(h)
    project_different_images(h, rvec, tvec)
    

def project_different_images(h_mat, rvec, tvec):
    
    camera_matrix = np.array(
            [
                [1024, 0, 1024],
                [0, 1024, 1024],
                [0,    0, 1],
            ],
            dtype='float32'
        )
    points_3d_world = np.array(
        [
            [-805465, 261912, 176528],
            [-999003, 271359, 176466],
            [-878350, 265396, 150586],
            [-903028, 266616, 150125],
            [-902612, 266558, 127889],
            [-878333, 265380, 127893],
        ],
    dtype='float32')
    img_paths = [
        r'C:\Users\HP\Downloads\360_5_separated_0606\pano_000005_000025_4.jpg',
        r'C:\Users\HP\Downloads\360_5_separated_0606\pano_000005_000026_4.jpg',
        r'C:\Users\HP\Downloads\360_5_separated_0606\pano_000005_000027_4.jpg',
        r'C:\Users\HP\Downloads\360_5_separated_0606\pano_000005_000028_4.jpg',
        r'C:\Users\HP\Downloads\360_5_separated_0606\pano_000005_000029_4.jpg',
        r'C:\Users\HP\Downloads\360_5_separated_0606\pano_000005_000030_4.jpg',
    ]
    povs = np.array(
        [
            [-837810, 184720, -4279], # 25    
            [-867970, 186820, -4739], # 26
            [-897550, 188829, -5380], # 27
            [-927669, 191009, -5639], # 28
            [-956849, 193179, -6370], # 29
            [-986809, 195610, -6210], # 30
            
        ],
        dtype='float32'
    )
    
    for i in range(len(povs)):
        pov = povs[i]
        img = cv2.imread(img_paths[i])
        
        points_3d_pov = points_3d_world.copy()
        for pt in points_3d_pov:
            pt -= pov
        
        # points_3d_cam = cv2.perspectiveTransform(points_3d_pov.reshape(-1, 1, 3), h_mat).reshape(-1, 3)
    
        points_2d = cv2.projectPoints(points_3d_pov, rvec, tvec, camera_matrix, None)[0].reshape(-1, 2)
        
        for pt in points_2d:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 15, (255, 0, 0), -1)
        cv2.imshow('img', cv2.resize(img, (600, 600)))
        cv2.waitKey()
    


def project_3d_to_flat(h):
    
    # img = np.ones((2048, 2048, 3), dtype='uint8')
    img = cv2.imread(r'C:\Users\HP\Downloads\360_5_separated_0606\pano_000005_000025_4.jpg')
    points_3d_world = np.array(
        [
            [-805465, 261912, 176528],
            [-999003, 271359, 176466],
            [-878350, 265396, 150586],
            [-903028, 266616, 150125],
            [-902612, 266558, 127889],
            [-878333, 265380, 127893],
        ],
    dtype='float32')   
    
    expected_pov = np.array([-837810, 184720, -4279], dtype='float32')
    for i in range(points_3d_world.shape[0]):
        points_3d_world[i] -= expected_pov
    
    points_3d_cam = cv2.perspectiveTransform(points_3d_world.reshape(-1, 1, 3), h).reshape(-1, 3)
    
    camera_matrix = np.array(
        [
            [1024, 0, 0],
            [0, 1024, 0],
            [0,    0, 1],
        ],
        dtype='float32'
    )
    points_2d = points_3d_cam @ camera_matrix
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]
    
    for pt in points_2d:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 15, (255, 0, 0), -1)
    cv2.imshow('img', cv2.resize(img, (400, 400)))
    cv2.waitKey()


def find_dif_h(h):
    hi = np.linalg.inv(h)
    expected_pov = np.array([[-837810, 184720, -4279]], dtype='float32')
    origin = np.zeros((1, 1, 3), dtype='float32')
    actual_pov = cv2.perspectiveTransform(origin, hi).reshape(-1, 3)
    dif = actual_pov - expected_pov
    print(actual_pov)
    print(dif)
    
    h[0:3, 3] += expected_pov[0]
    print(h)

    
def apply_h(h_mat):
    
    points_3d = np.array(
        [
            [-805465, 261912, 176528],
            [-999003, 271359, 176466],
            [-878350, 265396, 150586],
            [-903028, 266616, 150125],
            [-902612, 266558, 127889],
            [-878333, 265380, 127893],
        ],
    dtype='float32')   
     
    points_flat = np.array(
        [
            [1192, 602],
            [122, 635],
            [746, 539],
            [596, 545],
            [524, 469],
            [700, 460],
        ], 
    dtype='float32')
    
    pov = np.zeros((1, 3), dtype='float32')
    points_rays = np.concatenate([points_flat, 
                                  np.ones((points_flat.shape[0], 1), dtype='float32') * 1024], 
                                 axis=1) * 50
    
    points_3d2 = cv2.perspectiveTransform(points_3d.reshape(-1, 1, 3), h_mat).reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for pt in points_3d2:
        xs = [pov[0][0], pt[0]]
        ys = [pov[0][1], pt[1]] 
        zs = [pov[0][2], pt[2]]
        ax.plot(xs, ys, zs, color='#0000ff')

    xs = points_rays.T[0]
    ys = points_rays.T[1]
    zs = points_rays.T[2]
    ax.scatter(xs, ys, zs, 'o', color='#00ff00')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def solve_pnp():
    # pano25, flat 4
    points_flat = np.array(
        [
            [1192, 602],
            [122, 635],
            [746, 539],
            [596, 545],
            [524, 469],
            [700, 460],
        ], 
    dtype='float32')
    
    points_3d = np.array(
        [
            [-805465, 261912, 176528],
            [-999003, 271359, 176466],
            [-878350, 265396, 150586],
            [-903028, 266616, 150125],
            [-902612, 266558, 127889],
            [-878333, 265380, 127893],
        ],
    dtype='float32')
    
    camera_matrix = np.array(
        [
            [1024,   0, 0],
            [0,   1024, 0],
            [0,      0, 1],
        ],
    dtype='float32')
    
    retval, rvec, tvec = cv2.solvePnP(points_3d, points_flat, camera_matrix, None)
    print('rvec')
    print(rvec)
    print('tvec') 
    print(tvec)
    
    rot_mat = cv2.Rodrigues(rvec)[0]
    print('rot_mat')
    print(rot_mat)
    
    h_mat = np.eye(4, dtype='float32')
    h_mat[0:3, 0:3] = rot_mat
    h_mat[0:3, 3] = tvec.T
    
    print('h_mat')
    print(h_mat)
    
    return h_mat


def solve_pnp_pov():
    # pano25, flat 4
    points_flat = np.array(
        [
            [1192, 602],
            [122, 635],

            [746, 539],
            [596, 545],
            
            [524, 469],
            [700, 460],
        ], 
    dtype='float32')
    
    vis_img = cv2.imread(r'C:\Users\HP\Downloads\360_5_separated_0606\pano_000005_000025_4.jpg') #np.zeros((2048, 2048, 3), dtype='uint8')
    for pt in points_flat:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 15, (0, 0, 255), -1)
    cv2.imshow('img', cv2.resize(vis_img, (600, 600)))
    cv2.waitKey()
    
    points_3d = np.array(
        [
            [-805465, 261912, 176528],
            [-999003, 271359, 176466],
            
            [-878350, 265396, 150586],
            [-903028, 266616, 150125],
            
            [-902612, 266558, 127889],
            [-878333, 265380, 127893],
        ],
    dtype='float32')
    
    expected_pov = np.array([-837810, 184720, -4279], dtype='float32')
    for i in range(points_3d.shape[0]):
        points_3d[i] -= expected_pov
    
    camera_matrix = np.array(
        [
            [1024,   0, 1024],
            [0,   1024, 1024],
            [0,      0, 1],
        ],
    dtype='float32')
    
    retval, rvec, tvec = cv2.solvePnP(points_3d, points_flat, camera_matrix, None)
    print('rvec')
    print(rvec)
    print('tvec') 
    print(tvec)
    
    rot_mat = cv2.Rodrigues(rvec)[0]
    print('rot_mat')
    print(rot_mat)
    
    h_mat = np.eye(4, dtype='float32')
    h_mat[0:3, 0:3] = rot_mat
    h_mat[0:3, 3] = tvec.T
    
    print('h_mat')
    print(h_mat)
    
    return h_mat, rvec, tvec
    


def show_2d_point():
    img = cv2.imread(r'C:\Users\HP\Downloads\360_5\pano_000005_000025.jpg')
    points_pano = [
        [2491, 533], 
        [519, 974], 
        [1340, 634], 
        [1066, 715], 
        [1067, 807], 
        [1339, 721],
    ]
    
    for pt in points_pano:
        vis_img = img.copy()
        cv2.circle(vis_img, pt, 25, (0, 0, 255), -1)
        cv2.imshow('img', cv2.resize(vis_img, (600, 600)))
        cv2.waitKey()


if __name__ == '__main__':
    main()
