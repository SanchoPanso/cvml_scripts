import numpy as np
import cv2


import matplotlib.pyplot as plt
import numpy as np


# # pano 25
# points_pano = np.array(
#     [
#         [2491, 533], 
#         [519, 974], 
#         [1340, 634], 
#         [1066, 715], 
#         [1067, 807], 
#         [1339, 721],
        
#         # # new data 02.06
#         # [3949, 1999], [4108, 2003],
#         # [3946, 1922], [3962, 1922], [3946, 1963], [3962, 1963],
        
#         # #[2292, 1626], 
#         # [1654, 1611],
        
#         # [5050, 2189], [5446, 2219],
#     ], 
#     dtype='float64'
# )

# pano 26
points_pano = np.array([[2876, 626], [622, 863], [1814, 566], [1434, 607], [1430, 691], [1804, 648]], dtype='float64')

points_3d = np.array(
    [
        [-805465, 261912, 176528],
        [-999003, 271359, 176466],
        [-878350, 265396, 150586],
        [-903028, 266616, 150125],
        [-902612, 266558, 127889],
        [-878333, 265380, 127893],
        
        # # new data 02.06
        # [-32919, 160114,  12232], [-37525,  59656,  12465],
        # [ -6963, 164109,  61734], [ -7644, 150509,  61329], [ -7059, 164212,  37292], [ -7666, 151013,  37163],
        
        # #[-874860,  262376,   18930],
        # [-856556,  264411,   18594],
        
        # [-746847,   67831,  -21795], [-794234,   69906,  -21314],
    ]
)

povs = np.load(r'C:\Users\HP\Downloads\Telegram Desktop\points_of_view_5.npy')
points_of_view = povs

# difference = [-4347, 3548, 0]

# points_of_view = [
#     [-872317, 190368, -4739], # new 26 (?)
    
#     # [-842157, 188268, -4279], # new 25 (err = 70,37)
#     # [-841953, 188225, -4279],
#     # [-841303, 188143, -4279],
    
#     # [-837810, 184720, -4279], # 25    
#     # [-867970, 186820, -4739], # 26
# ]


#[[-897550, 188829, -5380]],
# # [[-927669, 191009, -5639]],

width = 8000
height = 4000
img = cv2.imread(r'C:\Users\HP\Downloads\Telegram Desktop\pano_000005_000026.jpg') #np.ones((height, wigth, 3), dtype='uint8') * 200


def main():
    project_3d_to_2d()
    project_2d_to_3d()
    find_pov()


def project_3d_to_2d():

    for i, point_of_view in enumerate(points_of_view):
        point_of_view = np.array(point_of_view).reshape(1, 3)
        print(i)
        print(point_of_view)
        # draw_3d_points_with_pov(points_3d, point_of_view)

        local_points_cart = points_3d - np.concatenate([point_of_view] * points_3d.shape[0], axis=0)
        local_points_cart = local_points_cart.astype('float64')
        #print(local_points_cart)

        local_points_sph = cart2sph(local_points_cart)
        local_points_projected = local_points_sph.copy()
        local_points_projected[:, 1] = local_points_projected[:, 1] / (np.pi) * height
        local_points_projected[:, 2] = width - local_points_projected[:, 2] / (2 * np.pi) * width

        n = len(local_points_projected)
        alpha = (points_pano[:, 0] - local_points_projected[:, 2]).sum() / n
        print(alpha)

        for i in range(local_points_sph.shape[0]):
            r, theta, phi = local_points_projected[i]
            x, y = points_pano[i]
            # cv2.circle(img, (int(phi), int(theta)), 20, (55, 0, 0), -1)
            cv2.circle(img, (int(x), int(y)), 25, (0, 0, 255), -1)
            cv2.circle(img, (int(phi + alpha), int(theta)), 25, (255, 0, 0), -1)


        cv2.imshow('img', cv2.resize(img, (800, 400)))
        cv2.waitKey()
    cv2.destroyAllWindows()


def project_2d_to_3d():
    pov = np.array(points_of_view[0]).reshape(1, 3)
    
    local_points_cart = points_3d - np.concatenate([pov] * points_3d.shape[0], axis=0)
    local_points_cart = local_points_cart.astype('float64')
    
    local_points_sph = cart2sph(local_points_cart)
    local_points_projected = local_points_sph.copy()
    local_points_projected[:, 1] = local_points_projected[:, 1] / (np.pi) * height
    local_points_projected[:, 2] = width - local_points_projected[:, 2] / (2 * np.pi) * width
    
    n = len(local_points_projected)    
    alpha = (points_pano[:, 0] - local_points_projected[:, 2]).sum() / n
    
    h = get_homography(alpha / width * 2 * np.pi, pov)

    rays = np.zeros((n, 3), dtype='float64')
    rays[:, 0] = 300*1000
    rays[:, 1] = points_pano[:, 1] / height * np.pi
    rays[:, 2] = 2 * np.pi - points_pano[:, 0] / width * 2 * np.pi


    rays_xyz = sph2cart(rays)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for ray in rays_xyz:
        pt = np.array([ray[0], ray[1], ray[2], 1])
        pt = h @ pt
        xs = [pov[0][0], pt[0]]
        ys = [pov[0][1], pt[1]] 
        zs = [pov[0][2], pt[2]]
        ax.plot(xs, ys, zs, color='#0000ff')

    xs = points_3d.T[0]
    ys = points_3d.T[1]
    zs = points_3d.T[2]
    ax.scatter(xs, ys, zs, marker='o', color='#ff0000')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def find_pov():
    lr = 2000
    dx = dy = dz = 10
    pov = [-842157, 188268, -4279]
    for i in range(2000):
        pov = np.array(pov).reshape(1, 3)
        err = get_error(pov)
        
        if i % 100 == 0:
            print(i)
            print('pov', pov)
            print('err', err)
            print()
        
        pov_dx = pov.copy()
        pov_dx[0][0] += dx
        err_dx = get_error(pov_dx)
        
        pov_dy = pov.copy()
        pov_dy[0][1] += dy
        err_dy = get_error(pov_dy)
        
        pov_dz = pov.copy()
        pov_dz[0][2] += dz
        err_dz = get_error(pov_dz)
        
        new_pov = pov.copy()
        new_pov[0][0] -= lr * (err_dx - err) / dx
        new_pov[0][1] -= lr * (err_dy - err) / dy
        new_pov[0][2] -= lr * (err_dz - err) / dz
        
        pov = new_pov    



def cart2sph(points: np.ndarray) -> np.ndarray:
    """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]    
    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x*x + y*y), z)
    phi = np.arctan2(y, x)
    
    return np.concatenate([[radius], [theta], [phi]], axis=0).T

def sph2cart(points: np.ndarray) -> np.ndarray:
    """Converts a spherical coordinate (radius, theta, phi) into a cartesian one (x, y, z)."""
    radius, theta, phi = points[:, 0], points[:, 1], points[:, 2]    
    x = radius * np.cos(phi) * np.sin(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(theta)
    
    return np.concatenate([[x], [y], [z]], axis=0).T



def get_homography(alpha: float, pov: np.ndarray):
  
    rot = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0, 0],
            [np.sin(alpha), np.cos(alpha), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype='float64'
    ) 
    shift = np.array(
        [
            [1, 0, 0, pov[0][0]],
            [0, 1, 0, pov[0][1]],
            [0, 0, 1, pov[0][2]],
            [0, 0, 0, 1],
        ],
        dtype='float64'
    )    
    return shift @ rot
    


def draw_3d_points_with_pov(points: np.ndarray, pov: np.ndarray = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs = points.T[0]
    ys = points.T[1]
    zs = points.T[2]
    ax.scatter(xs, ys, zs, marker='o')
    
    if pov is not None:
        xs = pov.T[0]
        ys = pov.T[1]
        zs = pov.T[2]
        ax.scatter(xs, ys, zs, marker='^')
        

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def get_error(pov: np.ndarray) -> float:
    
    local_points_cart = points_3d - np.concatenate([pov] * points_3d.shape[0], axis=0)
    local_points_cart = local_points_cart.astype('float64')
    
    local_points_sph = cart2sph(local_points_cart)
    local_points_projected = local_points_sph.copy()
    local_points_projected[:, 1] = local_points_projected[:, 1] / (np.pi) * height
    local_points_projected[:, 2] = width - local_points_projected[:, 2] / (2 * np.pi) * width
    
    n = len(local_points_projected)    
    alpha = (points_pano[:, 0] - local_points_projected[:, 2]).sum() / n
    err = ((points_pano[:, 0] - local_points_projected[:, 2] - alpha) ** 2).sum()
    return err


if __name__ == '__main__':
    main()
    
