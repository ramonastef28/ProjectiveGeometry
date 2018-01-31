# CS231A Homework 1, Problem 2
import numpy as np
from scipy import linalg
'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    # TODO: Fill in this code
    [sizeX, sizeY] = real_XY.shape
    z_front = np.zeros((sizeX,1))
    z_back = 150*np.ones((sizeX,1))
    A1 = np.concatenate((real_XY,z_front, np.ones((sizeX,1))),axis=1)
    b1=front_image[:,0]
    A2=np.concatenate((real_XY,z_back, np.ones((sizeX,1))),axis=1)
    b2=back_image[:,0]
    A=np.concatenate((A1,A2),axis=0)
    b=np.concatenate((b1,b2),axis=0)
    result_x=np.linalg.lstsq(A,b)[0]
    b_y=np.concatenate((front_image[:,1],back_image[:,1]),axis=0)
    result_y=np.linalg.lstsq(A,b_y)[0]
    camera_matrix=np.concatenate((result_x.T,result_y.T,np.array([0,0,0,1])),axis=0)

    return camera_matrix
'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    #TODO: Fill in this code
   
    [sizeX, sizeY] = real_XY.shape
    z_front = np.zeros((sizeX,1))
    z_back = 150*np.ones((sizeX,1))

    x = np.concatenate((front_image,back_image,np.ones((sizeX,sizeY))),axis=0)
    print(x)
    #x = np.concatenate((front_image,back_image,np.ones((sizeX,1))),axis=0)

    A1 = np.concatenate((real_XY,z_front, np.ones((sizeX,1))),axis=1)
    b1=front_image[:,0]
    A2=np.concatenate((real_XY,z_back, np.ones((sizeX,1))),axis=1)

    A=np.concatenate((A1,A2),axis=0)
    A=A.T 
    camera_matrix=np.reshape(camera_matrix,(3,4))
 
    x_pred = camera_matrix.dot(A)
    x_pred=x_pred[:2,:]
    x_image=np.concatenate((front_image,back_image),axis=0)
    error=x_pred-x_image.T
    error=np.multiply(error,error)
    error=np.sum(error,axis=0)
    error=np.sum(error/(error.shape[0]))
    rms_error = np.sqrt(error)
    return rms_error
if __name__ == '__main__':
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')



    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))
