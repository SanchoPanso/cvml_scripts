import numpy as np
import cv2


import matplotlib.pyplot as plt
import numpy as np


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
        dtype='float32'
    ) 
    # shift = np.array(
    #     [
    #         [1, 0, 0, pov[0][0]],
    #         [0, 1, 0, pov[0][1]],
    #         [0, 0, 1, pov[0][2]],
    #         [0, 0, 0, 1],
    #     ],
    #     dtype='float32'
    # )    
    return rot
    


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


width = 8000
height = 4000

# points_pano = np.array(
#     [[2491, 533], 
#      [519, 974], 
#      [1340, 634], 
#      [1066, 715], 
#      [1067, 807], 
#      [1339, 721]], dtype='float32')

points_pano = np.array([[2876, 626], [622, 863], [1814, 566], [1434, 607], [1430, 691], [1804, 648]], dtype='float32')

img = np.ones((height, width, 3), dtype='uint8') * 255
for i in range(points_pano.shape[0]):
    x, y = points_pano[i]
    cv2.circle(img, (int(x), int(y)), 25, (0, 0, 255), -1)

# cv2.imshow('src_img', cv2.resize(img, (400, 400)))
# cv2.waitKey()



points_3d = np.array(
    [
        [-805465, 261912, 176528],
        [-999003, 271359, 176466],
        [-878350, 265396, 150586],
        [-903028, 266616, 150125],
        [-902612, 266558, 127889],
        [-878333, 265380, 127893],
    ]
)

# povs = np.load(r'C:\Users\HP\Downloads\Telegram Desktop\points_of_view_5.npy')
# points_of_view = povs

points_of_view = [
    [-867970, 186820, -4739],
]


#[[-897550, 188829, -5380]],
    # [[-927669, 191009, -5639]],

for i, point_of_view in enumerate(points_of_view):
    point_of_view = np.array(point_of_view).reshape(1, 3)
    print(i)
    print(point_of_view)
    # draw_3d_points_with_pov(points_3d, point_of_view)

    local_points_cart = points_3d - np.concatenate([point_of_view] * points_3d.shape[0], axis=0)
    local_points_cart = local_points_cart.astype('float32')
    #print(local_points_cart)

    local_points_sph = cart2sph(local_points_cart)
    local_points_projected = local_points_sph.copy()
    local_points_projected[:, 1] = local_points_projected[:, 1] / (np.pi) * height
    local_points_projected[:, 2] = width - local_points_projected[:, 2] / (2 * np.pi) * width

    n = len(local_points_projected)
    alpha = (points_pano[:, 0] - local_points_projected[:, 2]).sum() / n
    #print(alpha)

    img = cv2.imread(r'C:\Users\HP\Downloads\Telegram Desktop\pano_000005_000026.jpg') #np.ones((height, wigth, 3), dtype='uint8') * 200
    cv2.line(img, (0, height // 2), (width, height // 2), (0, 255, 0), 3)


    for i in range(local_points_sph.shape[0]):
        r, theta, phi = local_points_projected[i]
        x, y = points_pano[i]
        cv2.circle(img, (int(phi), int(theta)), 20, (55, 0, 0), -1)
        cv2.circle(img, (int(phi + alpha), int(theta)), 20, (255, 0, 0), -1)
        cv2.circle(img, (int(x), int(y)), 25, (0, 0, 255), -1)


    cv2.imshow('img', cv2.resize(img, (800, 400)))
    cv2.waitKey()

h = get_homography(alpha / width * 2 * np.pi, point_of_view)

rays = np.zeros((n, 3), dtype='float32')
rays[:, 0] = 300*1000
rays[:, 1] = points_pano[:, 1] / height * np.pi
rays[:, 2] = 2 * np.pi - points_pano[:, 0] / width * 2 * np.pi


rays_xyz = sph2cart(rays)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for ray in rays_xyz:
    pt = np.array([ray[0], ray[1], ray[2], 1])
    pt = h @ pt
    xs = [0, pt[0]]
    ys = [0, pt[1]] 
    zs = [0, pt[2]]
    ax.plot(xs, ys, zs, color='#0000ff')

xs = points_3d.T[0] - point_of_view[0][0]
ys = points_3d.T[1] - point_of_view[0][1]
zs = points_3d.T[2] - point_of_view[0][2]
ax.scatter(xs, ys, zs, marker='o', color='#ff0000')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
