# Specify motion and force, display in aitviewer

import os

import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.spheres import Spheres


from aitviewer.renderables.lines import Lines
from aitviewer.renderables.arrows import Arrows


from aitviewer.forceplate import get_fp_asdict
import math as m
import torch

Basepath = "../GRF/SampleData/"
Forcepath = "../GRF/SampleData/"

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

PredPath = "../GRF/SampleData/"
participant = 'S7'
trial = 's007_20220705_hopping_1'
threshold = 0.3
sourcemotion = os.path.join(Basepath+participants_moshpp[participant], trial +'_stageii.npz')
fps = 250.0
Testing = True
ckp = 'noshape_s7_3e6_73_3e-6'
grf_file = os.path.join(Forcepath+participants_grf[participant], trial+'.npy')
if Testing:
    gt_file = os.path.join(PredPath+participant+'/test', trial+'.pth')
else:
    gt_file = os.path.join(PredPath+participant+'/preprocessed', trial+'.pth')
predicted = os.path.join(PredPath+participant+'/prediction/' + ckp, trial+'.pth')

import math as m

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

if __name__ == "__main__":
    # Load an AMASS sequence and make sure it's sampled at 60 fps. 
    # This loads the SMPL-X model.
    # We set transparency to 0.5 and render the joint coordinates systems.
    c = (149 / 255, 85 / 255, 149 / 255, 0.5)
    color_gt = (83 / 255, 189 / 255, 255 / 255, 1.0)
    color_pred = (255 / 255, 130 / 255, 53/255, 1.0)

    mesh = (102/255,102/255,102/255,0.5)
    fp_color = (127/255,127/255,128/255,1)
    seq_amass = SMPLSequence.from_amass(
        npz_data_path=sourcemotion,
        fps_out=fps,
        color=mesh,
        name=trial,
        show_joint_angles=True,
    )

    ptc_amass = PointClouds(seq_amass.vertices, position=np.array([-1.0, 0.0, 0.0]), color=c, z_up=True)

    # get force plates and lines 
    forceplate = get_fp_asdict(1,0,0,-0.5*m.pi,0,0)
    line_starts = forceplate['start']*0.01
    line_ends = forceplate['end']*0.01

    
    line_strip = np.zeros((2 * 20, 3))
    line_strip[::2] = line_starts
    line_strip[1::2] = line_ends
    line_renderable = Lines(line_strip, color = fp_color, mode="lines")


    gt_data = torch.load(gt_file)
    prediction = torch.load(predicted)
    CoP = prediction["CoP"]
    GRF = prediction["GRF"]


    CoP_pred = prediction["prediction"][..., :3]
    GRF_pred = prediction["prediction"][..., -3:]

    mask = GRF_pred[:,:,2] < threshold
    GRF_pred[mask] = 0


    transf_mat = gt_data["to_global"]
    homo = torch.ones(len(gt_data["CoP"]), 2, 1)
    CoP = torch.cat((CoP, homo), dim=-1)
    CoP_pred = torch.cat((CoP_pred, homo), dim=-1)

    
    CoP = torch.matmul(transf_mat, CoP.transpose(-1, -2)).transpose(-1, -2)[:, :, :-1]
    CoP_pred = torch.matmul(transf_mat, CoP_pred.transpose(-1, -2)).transpose(-1, -2)[:, :, :-1]


    CoP_pred = rotate_vis(CoP_pred)
    GRF_pred = rotate_vis(GRF_pred)
    CoP = rotate_vis(CoP)
    GRF = rotate_vis(GRF)
    print(GRF_pred)


    # print(drawpoint)
    pelvis_trans = gt_data["trans"]
    # print(pelvis_trans)
    pelvis = rotate_vis(pelvis_trans)
    # print(pelvis)
    # print(CoP)





    arrow_renderables = Arrows(
                CoP.numpy(),
                CoP.numpy()+GRF.numpy(),
                color= color_gt,
                is_selectable=True,
            )
    
    arrow_renderables_pred = Arrows(
                CoP_pred.numpy(),
                CoP_pred.numpy()+GRF_pred.numpy(),
                color= color_pred,
                is_selectable=True,
            )
    

    # Display in the viewer.
    v = Viewer()
    v.run_animations = True
    v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    # v.scene.add(seq_amass, ptc_amass)
    v.scene.add(seq_amass)

    v.scene.add(line_renderable)
    v.scene.add(arrow_renderables)
    v.scene.add(arrow_renderables_pred)

    v.run()
