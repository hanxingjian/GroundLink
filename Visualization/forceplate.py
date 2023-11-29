import numpy as np
import math as m
from collections import OrderedDict


def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def scale(points, factor):
    return points * factor


def rotate(data, axis, angle):
    if axis == 0:
        return np.array(np.matmul(Rx(angle), data.T).T)
    if axis == 1:
        return np.array(np.matmul(Ry(angle), data.T).T)
    if axis == 2:
        return np.array(np.matmul(Rz(angle), data.T).T)

def rotate_origin(data):
    # -1 * ground reaction force extracted from QTM
    return -1*data


def get_fp_coord(x, y, width, height, rotationX, rotationY, rotationZ):
    up_left = [x+width, y+height, 0]
    up_right = [x, y+height, 0]
    bot_left = [x+width, y, 0]
    bot_right = [x,y, 0]
    start = [bot_right,bot_right,up_left,up_left]
    end = [bot_left, up_right, up_right, bot_left]
    # print(start)
    # print(end)
    
    rot_start_x = rotate(np.asarray(start), 0, rotationX).tolist()
    rot_end_x = rotate(np.asarray(end), 0, rotationX).tolist()


    rot_start = rotate(np.asarray(rot_start_x), 1, rotationY).tolist()
    rot_end = rotate(np.asarray(rot_end_x), 1, rotationY).tolist()

    
    return rot_start, rot_end


def get_fp_asdict(scale, shift_x, shift_y, rotationX, rotationY, rotationZ):
    width = -40*scale
    height = 60*scale
    shift_x *= scale
    shift_y *= scale
    fp1start, fp1end = get_fp_coord(0+shift_x,0+shift_y,width,height,rotationX, rotationY, rotationZ)
    fp2start, fp2end = get_fp_coord(0+shift_x,60*scale+shift_y,width,height,rotationX, rotationY, rotationZ)
    fp3start, fp3end = get_fp_coord(0+shift_x,120*scale+shift_y,width,height,rotationX, rotationY, rotationZ)
    fp4start, fp4end = get_fp_coord(-40*scale+shift_x,30*scale+shift_y,width,height,rotationX, rotationY, rotationZ)
    fp5start, fp5end = get_fp_coord(-40*scale+shift_x,90*scale+shift_y,width,height,rotationX, rotationY, rotationZ)
    force_plate = OrderedDict()
    force_plate['start'] = np.asarray(fp1start+fp2start+fp3start+fp4start+fp5start)
    force_plate['end'] = np.asarray(fp1end+fp2end+fp3end+fp4end+fp5end)
    return force_plate