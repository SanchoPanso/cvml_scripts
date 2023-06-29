import os
import numpy as np
import cv2
import glob
import open3d as o3d
import json
import pandas as pd
from ultralytics import YOLO
from sklearn.neighbors import KDTree
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM


def main():
    class_names = ['trees', 'animal farms', 'signboard', 'garbage', 'buildings', 'light poles', 'traffic sign']
    
    cal_path = r'C:\Users\HP\Downloads\Telegram Desktop\MX9_Dual.Extcal.json'
    trajectory_path = r'C:\Users\HP\Downloads\Telegram Desktop\reference.csv'
    points_path = r'D:\CodeProjects\PythonProjects\cvml_scripts\geo_ai\down_pcd_points_global.npy'
    img_dir = r'D:\geo_ai_data\shots\5'
    model_path = r'C:\Users\HP\Downloads\yolov8s_seg_geoai360_07062023_2_best.pt'
    
    points = np.load(points_path)
    
    trajectory = pd.read_csv(trajectory_path, sep='\t')
    trajectory.index = trajectory['file_name'].tolist()
    model = YOLO(model_path)
    
    installation = get_installation(cal_path)
    
    template_img_name = 'pano_000005_{0:06d}'
    start = 10
    stop = 45
    
    common_mask = np.zeros((len(points),), dtype='bool')
    
    object_pos = []
    for i in range(start, stop):
        for j in range(4):
            orig_img_fn = template_img_name.format(i)
            img_fn = orig_img_fn + f'_{j}.jpg'
            print(img_fn)
            
            img = cv2.imread(os.path.join(img_dir, img_fn))
            
            result = model(img)[0]
            boxes = result.boxes 
            masks = result.masks
            
            for k, mask in enumerate(masks.data):
                if class_names[int(boxes.cls[k].item())] != 'trees':
                    continue
                
                x, y, w, h = map(lambda x: x.item() * 640, boxes.xywhn[k])
                if w < 150 or h < 150:
                    continue
                
                mask = mask.cpu().numpy().astype('uint8')
                mask = cv2.erode(mask, (10, 10))
                # # cv2.imshow('mask', mask * 255)
                # # cv2.waitKey()
                mask_bottom = mask.copy()
                mask_bottom[int(y - 0.5 * h): int(y + 0.4 * h), 
                            int(x - 0.5 * w): int(x + 0.6 * w)] = 0
                # # cv2.imshow('mask', mask * 255)
                # # cv2.waitKey()
                
                cur_traectory_info = trajectory.loc[orig_img_fn]
                pos = cur_traectory_info[
                    [
                        'projectedX[m]', 
                        'projectedY[m]', 
                        'projectedZ[m]',
                    ]
                ].to_numpy()
                
                rotation = cur_traectory_info[
                    [
                        'roll[deg]',
                        'pitch[deg]',
                        'heading[deg]',
                    ]
                ].to_numpy()
                
                # print(pos)
                # print(rotation)
                # print(installation)
                
                target_mask = find_target_mask(
                    points,
                    mask,
                    pos,
                    rotation,
                    get_rot_oy(-j * np.pi / 2) @ installation,
                )
                
                # target_mask_bottom = find_target_mask(
                #     points,
                #     mask_bottom,
                #     pos,
                #     rotation,
                #     get_rot_oy(-j * np.pi / 2) @ installation,
                # )
                
                # common_mask |= target_mask
                # common_mask_bottom |= target_mask_bottom
                
                if target_mask.max() == False:
                    continue
                
                target_points = points[target_mask]
                tree = KDTree(target_points, leaf_size=max(1, int(0.75 * len(target_points))))
                num_nearest_pts = min(20, len(target_points))
                nearest_pts_ids = tree.query(pos.reshape(1, -1), num_nearest_pts, sort_results=True)[1][0]
                # nearest_pts = target_points[nearest_pts_ids]
                
                target_nearest_mask = np.zeros((len(points),), dtype='bool')
                tmp_mask = target_nearest_mask[target_mask]
                tmp_mask[nearest_pts_ids] = True
                target_nearest_mask[target_mask] = tmp_mask
                
                if num_nearest_pts > 1:
                    # estimator = LocalOutlierFactor(n_neighbors=min(num_nearest_pts, 5), contamination=0.15)
                    # outlier_pred = estimator.fit_predict(points[target_nearest_mask])
                    
                    # estimator =  SGDOneClassSVM(
                    #                 nu=0.15,
                    #                 shuffle=True,
                    #                 fit_intercept=True,
                    #                 random_state=42,
                    #                 tol=1e-6,)
                    # outlier_pred = estimator.fit(points[target_nearest_mask]).predict(points[target_nearest_mask])
                    
                    estimator =  EllipticEnvelope(contamination=0.15, random_state=42)
                    outlier_pred = estimator.fit(points[target_nearest_mask]).predict(points[target_nearest_mask])
                    
                    target_inliers_mask = np.zeros((len(points),), dtype='bool')
                    tmp_mask = target_inliers_mask[target_nearest_mask]
                    tmp_mask = (outlier_pred == 1)
                    target_inliers_mask[target_nearest_mask] = tmp_mask
                    
                    common_mask |= target_inliers_mask
                    
                    target_inliers = points[target_inliers_mask]
                    object_pos.append([target_inliers[:, 0].mean(),
                                    target_inliers[:, 1].mean(),
                                    target_inliers[:, 2].mean()])
                else:
                    common_mask |= target_nearest_mask
                
    
    #target_points = points[common_mask & (~common_mask_bottom)]
    #target_bottom_points = points[common_mask_bottom]
    target_points = points[common_mask]
    background_points = points[~common_mask]
    
    # np.save('target_points.npy', target_points)
    # np.save('target_bottom_points.npy', target_bottom_points)
    # np.save('background_points.npy', background_points)

    target_pcd = o3d.geometry.PointCloud()
    background_pcd = o3d.geometry.PointCloud()
    obj_pcd = o3d.geometry.PointCloud()

    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    target_pcd.paint_uniform_color([1, 0, 0])
    
    # target_bottom_pcd.points = o3d.utility.Vector3dVector(target_bottom_points)
    # target_bottom_pcd.paint_uniform_color([1, 1, 0])

    background_pcd.points = o3d.utility.Vector3dVector(background_points)
    colors = np.zeros(background_points.shape, dtype='float32')
    colors[:, 1] = (background_points[:, 2] - background_points[:, 2].min()) / (background_points[:, 2].max() - background_points[:, 2].min())
    colors[:, 2] = 1 - colors[:, 1]
    background_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    obj_pcd.points = o3d.utility.Vector3dVector(np.array(object_pos))
    obj_pcd.paint_uniform_color([1, 1, 0])
    
    # povs_pcd.points = o3d.utility.Vector3dVector(povs)
    # povs_pcd.paint_uniform_color([1, 1, 0])

    o3d.visualization.draw_geometries([target_pcd, background_pcd, obj_pcd])


class ObjectPoints:
    pos_ids: np.ndarray
    pts_ids: np.ndarray
    masks: np.ndarray
    
    def __init__(self, pos_ids: np.ndarray, pts_ids: np.ndarray, masks: np.ndarray):
        self.pos_ids = pos_ids
        self.pts_ids = pts_ids
        self.masks = masks
    
    def __add__(self, other):
        pos_ids = np.concatenate([self.pos_ids, other.pos_ids], axis=0)
        pts_ids = np.concatenate([self.pts_ids, other.pts_ids], axis=0)
        masks = np.concatenate([self.masks, other.masks], axis=0)
        
        return ObjectPoints(pos_ids, pts_ids, masks)
    
    def get_center(self) -> np.ndarray:
        pass


def find_target_mask(
    points: np.ndarray, 
    image_mask: np.ndarray, 
    pos: np.ndarray,
    rotation: np.ndarray,
    installation: np.ndarray) -> np.ndarray:
    
    cube_length = image_mask.shape[0]
    camera_matrix = np.array(
        [
            [cube_length / 2,                0, cube_length / 2],
            [              0,  cube_length / 2, cube_length / 2],
            [              0,                0,               1],
        ],
        dtype='float32'
    )

    r_rad = rotation[0] * np.pi / 180
    p_rad = rotation[1] * np.pi / 180
    h_rad = (rotation[2] - 90) * np.pi / 180   # "-90" - emperical found offset to turn heading into car direction
    
    # Minus because of inverse order of operations
    roll_rot = get_rot_ox(-r_rad)
    pitch_rot = get_rot_oy(-p_rad)
    heading_rot = get_rot_oz(-h_rad)

    trans = get_translation(pos[0], pos[1], pos[2])

    # Extrinsic matrix that maps into car coordinate system
    extrinsic_car_inv = trans @ heading_rot @ pitch_rot @ roll_rot
    extrinsic_car = np.linalg.inv(extrinsic_car_inv)
    
    # Extrinsic matrix that maps into camera coordinate system
    extrinsic_cam = installation @ extrinsic_car

    # Extrinsic matrix in form of rvec (Rodrigues form of rotation) and tvec (translation vector)
    rvec = cv2.Rodrigues(extrinsic_cam[0:3, 0:3])[0]
    tvec = extrinsic_cam[0:3, 3:4]
    
    # Remove hidden points, that is leave only visible points from camera position
    camera_position = np.linalg.inv(extrinsic_cam)[:3, 3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    _, pt_map = pcd.hidden_point_removal(camera_position, radius=1000)
    pcd_sel = pcd.select_by_index(pt_map)
    hpr_points = np.asarray(pcd_sel.points)
    
    # Turn world coordinates into camera coordinates.
    points_cam = cv2.perspectiveTransform(hpr_points.reshape(-1, 1, 3), extrinsic_cam).reshape(-1, 3)
    
    # Create mask to hide point, that places in non-visible half of space
    # If it is not done, then points from non-visible half will be projected on result image
    visibility_mask = points_cam[:, 2] > 0
    visible_points_cam = points_cam[visibility_mask]
    
    # Project 3d point in camera coord system onto image surface
    image_points, jacobian = cv2.projectPoints(
        visible_points_cam.reshape(-1, 1, 3),
        np.zeros((1, 3), dtype='float32'),
        np.zeros((1, 3), dtype='float32'),
        camera_matrix,
        None,
    )
    image_points = image_points.reshape(-1, 2)
    image_points = image_points.astype('int32')
    
    # Create mask, that hides out of bounds image points
    bounding_mask = (image_points[:, 0] >= 0) & (image_points[:, 0] < image_mask.shape[1]) & \
                    (image_points[:, 1] >= 0) & (image_points[:, 1] < image_mask.shape[0])
    image_points_in_bounds = image_points[bounding_mask]

    # Create target mask to define points, that correspond to target object from image mask
    target_mask = image_mask[image_points_in_bounds[:, 1], image_points_in_bounds[:, 0]] != 0
    image_points_in_bounds = image_points[bounding_mask]
    
    # Create common_target_mask for all original points (after hidden points removal),
    # where True - target, False - background
    common_target_mask = np.zeros((len(points),), dtype='bool')
    common_target_mask[pt_map] = True
    common_target_mask[common_target_mask] = visibility_mask
    common_target_mask[common_target_mask] = bounding_mask
    common_target_mask[common_target_mask] = target_mask
    
    return common_target_mask


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


def get_installation(cal_path):
    with open(cal_path) as f:
        data = json.load(f)
        installation_matrix_str = data['System']['Sensors'][-1]['Calibrations'][-1]['InstallationMatrix']
        refinement_matrix_str = data['System']['Sensors'][-1]['Calibrations'][-1]['RefinementMatrix']
            
    installation_matrix = np.fromstring(installation_matrix_str, sep=' ').reshape(4, 4)
    refinement_matrix = np.fromstring(refinement_matrix_str, sep=' ').reshape(4, 4)
    
    # Original installation matrix use unknown car and camera coordinate systems,
    # which different from chosen for this work.
    # That's why we need to "fix" it by applying different rotations
    y_rot90 = get_rot_oy(np.pi / 2)
    x_rot90 = get_rot_ox(np.pi / 2)
    y_rot_opt_axis = get_rot_oy(np.pi * 0.4)

    fixed_installation_matrix = x_rot90 @ y_rot90 @                       \
                                y_rot_opt_axis.T @                        \
                                refinement_matrix @ installation_matrix @ \
                                y_rot_opt_axis @                          \
                                x_rot90 @ y_rot90

    
    return get_rot_oy(np.pi) @ fixed_installation_matrix


if __name__ == '__main__':
    main()
