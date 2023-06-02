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
    shift = np.array(
        [
            [0, 0, 0, 0],
        ]
    )    
    


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

points_pano = np.array(
    [[2491, 533], 
     [519, 974], 
     [1340, 634], 
     [1066, 715], 
     [1067, 807], 
     [1339, 721]], dtype='float32')


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

point_of_view = np.array([[-927669, 191009, -5639]])

#[[-897550, 188829, -5380]],
    # [[-927669, 191009, -5639]],

while True:
    print(point_of_view)
    # draw_3d_points_with_pov(points_3d, point_of_view)

    local_points_cart = points_3d - np.concatenate([point_of_view] * points_3d.shape[0], axis=0)
    local_points_cart = local_points_cart.astype('float32')
    print(local_points_cart)

    local_points_sph = cart2sph(local_points_cart)
    local_points_projected = local_points_sph.copy()
    local_points_projected[:, 1] = local_points_projected[:, 1] / (np.pi) * height
    local_points_projected[:, 2] = local_points_projected[:, 2] / (2 * np.pi) * width

    n = len(local_points_projected)
    alpha = (points_pano[:, 0] - local_points_projected[:, 2]).sum() / n
    print(alpha)

    img = cv2.imread(r'C:\Users\HP\Downloads\Telegram Desktop\pano_000005_000025.jpg') #np.ones((height, wigth, 3), dtype='uint8') * 200
    cv2.line(img, (0, height // 2), (width, height // 2), (0, 255, 0), 3)


    for i in range(local_points_sph.shape[0]):
        r, theta, phi = local_points_projected[i]
        x, y = points_pano[i]
        cv2.circle(img, (int(phi + alpha), int(theta)), 20, (255, 0, 0), -1)
        cv2.circle(img, (int(x), int(y)), 25, (0, 0, 255), -1)


    cv2.imshow('img', cv2.resize(img, (800, 400)))
    p = cv2.waitKey()
    if p == ord('w'):
        point_of_view[0][1] += 1000
    elif p == ord('s'):
        point_of_view[0][1] -= 1000
    elif p == ord('a'):
        point_of_view[0][0] -= 1000
    elif p == ord('d'):
        point_of_view[0][0] += 1000
    elif p == ord('e'):
        point_of_view[0][2] += 1000
    elif p == ord('q'):
        point_of_view[0][2] -= 1000
    elif p == 27:
        break


