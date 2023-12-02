import numpy as np
import math as m

import torch

from aitviewer.forceplate import get_fp_asdict


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

def rotate(data, axis, angle):
    if axis == 0:
        return np.array(np.matmul(Rx(angle), data.T).T)
    if axis == 1:
        return np.array(np.matmul(Ry(angle), data.T).T)
    if axis == 2:
        return np.array(np.matmul(Rz(angle), data.T).T)

def rotate_vis(data):
   return torch.matmul(torch.tensor(Rx(-0.5*m.pi)), data.transpose(-1, -2).type('torch.DoubleTensor')).transpose(-1, -2)



participants_grf = {
    "S1" : "s001_20220513",
    "S2" : "s002_20220520",
    "S3" : "s003_20220523",
    "S4" : "s004_20220524",
    "S5" : "s005_20220610",
    "S6" : "s006_20220614",
    "S7" : "s007_20220705",
	}

participants_moshpp = {
    "S1" : "QTM_s001",
    "S2" : "QTM_s002",
    "S3" : "QTM_s003",
    "S4" : "QTM_s004",
    "S5" : "QTM_s005",
    "S6" : "QTM_s006",
    "S7" : "QTM_s007",
	}

participants = {
    "s001" : "S1",
    "s002" : "S2",
    "s003" : "S3",
    "s004" : "S4",
    "s005" : "S5",
    "s006" : "S6",
    "s007" : "S7",
	}


def get_fp():
  # get force plates and lines 
  forceplate = get_fp_asdict(1,0,0,-0.5*m.pi,0,0)
  line_starts = forceplate['start']*0.01
  line_ends = forceplate['end']*0.01

  
  line_strip = np.zeros((2 * 20, 3))
  line_strip[::2] = line_starts
  line_strip[1::2] = line_ends

  return line_strip


def get_data_pred(gt_file, predicted, threshold):
  gt_data = torch.load(gt_file)
  prediction = torch.load(predicted)
  CoP = prediction["CoP"]
  GRF = prediction["GRF"]


  CoP_pred = prediction["prediction"][..., :3]
  GRF_pred = prediction["prediction"][..., -3:]

  mask = GRF_pred[:,:,2] < threshold
  GRF_pred[mask] = 0


  transf_mat = gt_data["to_global"]
  # rot_mat = gt_data["to_global_rot"]
  homo = torch.ones(len(gt_data["CoP"]), 2, 1)
  CoP = torch.cat((CoP, homo), dim=-1)
  CoP_pred = torch.cat((CoP_pred, homo), dim=-1)

  
  CoP = torch.matmul(transf_mat, CoP.transpose(-1, -2)).transpose(-1, -2)[:, :, :-1]
  CoP_pred = torch.matmul(transf_mat, CoP_pred.transpose(-1, -2)).transpose(-1, -2)[:, :, :-1]
  # GRF = torch.matmul(rot_mat, GRF.type('torch.DoubleTensor').transpose(-1, -2)).transpose(-1, -2)
  # GRF_pred = torch.matmul(rot_mat, GRF_pred.type('torch.DoubleTensor').transpose(-1, -2)).transpose(-1, -2)


  CoP_pred = rotate_vis(CoP_pred)
  GRF_pred = rotate_vis(GRF_pred)
  CoP = rotate_vis(CoP)
  GRF = rotate_vis(GRF)

  return CoP, CoP_pred, GRF, GRF_pred