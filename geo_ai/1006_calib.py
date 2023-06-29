import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import List


def main():
    # pano_img = cv2.imread(r'C:\Users\HP\Downloads\360_5\pano_000005_000025.jpg')
    # flat_imgs = pano2flat(pano_img)
    # for i, img in enumerate(flat_imgs):
    #     cv2.imshow(str(i), cv2.resize(img, (400, 400)))
    # cv2.waitKey()
    
    # img1 = cv2.imread(r'C:\Users\HP\Downloads\360_5_separated_0606\pano_000005_000026_1.jpg')[:1024]
    # img2 = cv2.imread(r'C:\Users\HP\Downloads\360_5_separated_0606\pano_000005_000025_1.jpg')[:1024]
    
    # match_images(img1, img2)
    
    do_manual_calibration()


def pano2flat(pano: np.ndarray, 
              img_size: tuple = (2048, 2048), 
              f: float = 1024,
              pixels_per_radian = 4000 / np.pi) -> List[np.ndarray]:
    
    width, height = img_size

    x = np.arange(-width // 2, width // 2, 1, dtype='float32').reshape(1, width)
    y = np.arange(-height // 2, height // 2, 1, dtype='float32').reshape(1, height)
    
    flat_imgs = []        
    for i in range(4):
        c_phi =  i * np.pi / 2
        c_theta =  np.pi / 2

        map_x_1d = (c_phi + np.arctan(x / f))
        map_x  = np.ones((height, 1), dtype='float32') @ map_x_1d * pixels_per_radian
        map_y = (c_theta + np.arctan(y.T @ (np.cos(map_x_1d - c_phi)) / f)) * pixels_per_radian

        lateral_img = cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        flat_imgs.append(lateral_img)
        
    for i in range(2):
        c_phi =  np.pi
        c_theta =  i * np.pi
        
        x_2d = np.ones((height, 1), dtype='float32') @ x
        y_2d =  y.reshape(height, 1) @ np.ones((1, width), dtype='float32')

        map_x  = (c_phi + np.arctan2(y_2d, x_2d)) * pixels_per_radian 
        map_y = (c_theta + np.arctan(np.sqrt((x_2d ** 2 + y_2d ** 2)) / f)) * pixels_per_radian

        vertical_img = cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        flat_imgs.append(vertical_img)
    
    return flat_imgs


def do_manual_calibration():
    object_points = np.array(
        [
            [-805465, 261912, 176528],
            [-999003, 271359, 176466],
            
            [-878350, 265396, 150586],
            [-903028, 266616, 150125],
            
            [-902612, 266558, 127889],
            [-878333, 265380, 127893],
        ],
    dtype='float32')   
     
    image_points = np.array(
        [
            [1192, 602],
            [122, 635],

            [746, 539],
            [596, 545],
            
            [524, 469],
            [700, 460],
        ],  
    dtype='float32')
    
    camera_matrix = np.array(
        [
            [1024,   0, 1024],
            [0,   1024, 1024],
            [0,      0,    1],
        ],
    dtype='float32')
    
    # Find extrisic calibration
    retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, None)
    
    # Vizualize image points pojection to 3d space
    h_mat = get_homography(rvec, tvec)
    ray_points = np.ones((len(image_points), 3), dtype='float32') * 1024
    ray_points[:, 0:2] = image_points
    ray_points[:, 0] -= 1024
    ray_points[:, 1] -= 1024
    ray_points *= 200
    object_points_cam = cv2.perspectiveTransform(object_points.reshape(-1, 1, 3), h_mat).reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for pt in ray_points:
        xs = [0, pt[0]]
        ys = [0, pt[1]]
        zs = [0, pt[2]]
        ax.plot(xs, ys, zs, color='#0000ff')

    xs = object_points_cam.T[0]
    ys = object_points_cam.T[1]
    zs = object_points_cam.T[2]
    ax.scatter(xs, ys, zs, 'o', color='#00ff00')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
    # Vis
    img = np.zeros((2048, 2048, 3), dtype='uint8')
    res_image_points = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)[0].reshape(-1, 2)
    
    for pt in image_points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 15, (0, 255, 0), -1)
    for pt in res_image_points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 15, (0, 0, 255), -1)
        
    cv2.imshow('img', cv2.resize(img, (600, 600)))
    cv2.waitKey()
    
    return rvec, tvec


def match_images(img1: np.ndarray, img2: np.ndarray):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)], True, (0, 255, 255), 3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3[:, :, ::-1], 'gray'),plt.show()
    
    return M


def get_homography(rvec, tvec):
    h_mat = np.eye(4, dtype='float32')
    rot_mat = cv2.Rodrigues(rvec)[0]
    h_mat[0:3, 0:3] = rot_mat
    h_mat[0:3, 3:4] = tvec
    
    return h_mat


if __name__ == '__main__':
    main()
