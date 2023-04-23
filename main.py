import cv2
import numpy as np
import matplotlib.pyplot as plt

#######################################################################
## x Rotation Matrix
def x_rot_mat(x_theta):
    return np.array([[1, 0, 0], [0, np.cos(x_theta), -np.sin(x_theta)], [0, np.sin(x_theta), np.cos(x_theta)]])

## y Rotation Matrix
def y_rot_mat(y_theta):
    return np.array([[np.cos(y_theta), 0, np.sin(y_theta)], [0, 1, 0], [-np.sin(y_theta), 0, np.cos(y_theta)]])

## z Rotation Matrix
def z_rot_mat(z_theta):
    return np.array([[np.cos(z_theta), -np.sin(z_theta), 0], [np.sin(z_theta), np.cos(z_theta), 0], [0, 0, 1]])

## cv2 window init and create trackba
def cv2_init():    
    cv2.namedWindow("ROT CUBE")
    
    cv2.createTrackbar("length", "ROT CUBE", 1, 40, lambda x:x)
    cv2.createTrackbar("x_theta", "ROT CUBE", 0, 360, lambda x:x)
    cv2.createTrackbar("y_theta", "ROT CUBE", 0, 360, lambda x:x)
    cv2.createTrackbar("z_theta", "ROT CUBE", 0, 360, lambda x:x)
#######################################################################


def main():    
    ## Cube coordinates with 8 point
    ## Coordinates based on origin
    base_coords = np.array([[-1, -1 , 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]])

    ## Draw line index
    line_idx = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7], [4, 7]]

    ## Set line color
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(line_idx))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]


    
    ## Run
    while cv2.waitKey(1) != ord('q'):
        bg = np.zeros((500, 1000, 3)).astype(np.uint8) # Background
        h, w, c = bg.shape
        center = (w//2, h//2)
    
        ## cube coordinates based on origin with (range -1 ~ 1) * length
        length = cv2.getTrackbarPos("length", "ROT CUBE")
        coords = base_coords * length
    
        ## Multiply Rotation Matrix of each x,y,z axis
        x_theta = np.deg2rad(cv2.getTrackbarPos("x_theta", "ROT CUBE"))
        y_theta = np.deg2rad(cv2.getTrackbarPos("y_theta", "ROT CUBE"))
        z_theta = np.deg2rad(cv2.getTrackbarPos("z_theta", "ROT CUBE"))
        xyz_rot_coords = coords @ z_rot_mat(z_theta) @ y_rot_mat(y_theta) @ x_rot_mat(x_theta) 
        '''
        z_rot_coords = np.matmul(coords, z_rot_mat(z_theta))
        yz_rot_coords = np.matmul(z_rot_coords, y_rot_mat(y_theta))
        xyz_rot_coords = np.matmul(yz_rot_coords, x_rot_mat(x_theta))
        '''
        
        ## Orthogonal projection / remove z axis
        xy_axis_coords = xyz_rot_coords[...,:2] 
        ## Move to background center and type conversion as int
        xy_axis_trans_coords = (xy_axis_coords + center).astype(np.int32)
    
    
        ## draw circle
        for n, coord in enumerate(xy_axis_trans_coords):
            bg = cv2.circle(bg, coord, 3, colors[n], -1)
        ## draw line
        for n, idx in enumerate(line_idx):
            bg = cv2.line(bg, xy_axis_trans_coords[idx[0]], xy_axis_trans_coords[idx[1]], colors[n], 1)
    
        cv2.imshow("ROT CUBE", bg)
    
    
    
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cv2_init()
    main()