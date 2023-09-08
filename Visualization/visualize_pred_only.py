# Specify motion and force, display in aitviewer

import os

import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

from aitviewer.renderables.lines import Lines
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.spheres import Spheres


from aitviewer.forceplate import get_fp_asdict
import math as m
import torch

Basepath = "/home/xjhan/Documents/Research/Data/AMASS/ACCAD/"
# Basepath = "/home/xjhan/Documents/Research/Data/AMASS/Transitions/"

Forcepath = "../data/qtm/grf/"
PredPath = "../../dataset/no_shape/"
seq = "Male2MartialArtsKicks_c3d"

ckp = 'noshape_s7_3e6_73_3e-6'

participant = 'AMASS' + '/' + seq
trial = 'G3_-_front_kick'
# twistdance_jumpingtwist360
#  trial[:4]+'/'+
sourcemotion = os.path.join(Basepath+seq, trial +'_stageii.npz')
fps = 120.0

# grf_file = os.path.join(Forcepath+participants_grf[participant], trial+'.npy')
gt_file = os.path.join(PredPath+participant+'/preprocessed', trial+'.pth')
predicted = os.path.join(PredPath+participant+'/prediction/'+ckp, trial+'.pth')

# denormalized = os.path.join(PredPath+participant+'/postprocessed', trial+'.pth')
# print(denormalized)

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

test = '../data/amass/test_s004shape.npz'

if __name__ == "__main__":
    # Load an AMASS sequence and make sure it's sampled at 60 fps. This automatically loads the SMPL-H model.
    # We set transparency to 0.5 and render the joint coordinates systems.
    c = (149 / 255, 85 / 255, 149 / 255, 0.5)
    color_gt = (0 / 255, 191 / 255, 255 / 255, 1.0)
    color_pred = (255 / 255, 165 / 255, 0.0, 1.0)
    mesh = (102/255,102/255,102/255,0.6)
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

    
    # line_ends[:, 1] = 1.0
    line_strip = np.zeros((2 * 20, 3))
    line_strip[::2] = line_starts
    line_strip[1::2] = line_ends
    line_renderable = Lines(line_strip, mode="lines")

    # add GRF data
    # arrow: start (CoP), end (GRF)
    # grf_data = np.load(grf_file, allow_pickle=True)
    # CoP = grf_data.item()["CoP"]
    # GRF = grf_data.item()["GRF"]



    gt_data = torch.load(gt_file)


    # CoP_z = torch.zeros(len(CoP), 2, 1)
    # CoP = torch.index_select(torch.cat((CoP, CoP_z), dim=2), 2, torch.LongTensor([0,2,1]))

    
    pred_data = torch.load(predicted)
    CoP_pred = pred_data[..., :3]
    GRF_pred = pred_data[..., -3:]
    # CoP_pred = torch.index_select(torch.cat((CoP_pred, CoP_z), dim=2), 2, torch.LongTensor([0,2,1]))
    print(GRF_pred)
    
    


    # shift back to Ground
    pelvis_trans = gt_data["trans"]
    # print(pelvis_trans)
    # CoP_pred = CoP_pred+pelvis_trans

    somepoint = torch.tensor([0,0,0, 1])
    vec = torch.tensor([0,0,0.8])
    somepoint = somepoint.repeat(len(gt_data["to_global"]),1, 1)
    vec = vec.repeat(len(gt_data["to_global"]),1, 1)



    # pelvis_trans = gt_data["trans"]
    # pelvis_rot = gt_data["pelvis_rot_mat"]
    # CoP_pred = torch.matmul(pelvis_rot, CoP_pred.transpose(-1, -2).type('torch.DoubleTensor')).transpose(-1, -2) + pelvis_trans
    
    transf_mat = gt_data["to_global"]
    # rot_mat = gt_data["to_global_rot"]
    
    homo = torch.ones(len(gt_data["to_global"]), 2, 1)
    CoP_pred = torch.cat((CoP_pred, homo), dim=-1)
    # GRF_pred = torch.cat((GRF_pred, homo), dim=-1)

    # print(transf_mat[0])
    # print(transf_mat[600])

    
    # print(somepoint.size())
    drawpoint = torch.matmul(transf_mat, somepoint.type('torch.FloatTensor').transpose(-1, -2)).transpose(-1, -2)[:, :, :-1]
    # print(drawpoint)

    
    CoP_pred = torch.matmul(transf_mat, CoP_pred.transpose(-1, -2)).transpose(-1, -2)[:, :, :-1]
    # GRF_pred = torch.matmul(rot_mat, GRF_pred.transpose(-1, -2)).transpose(-1, -2)[:, :, :-1]
    # rot_mat = gt_data["to_global_rot"]
    # GRF_pred = torch.matmul(rot_mat, GRF_pred.type('torch.DoubleTensor').transpose(-1, -2)).transpose(-1, -2)

    threshold = 0.12
    # GRF_pred[:, :, 2] = torch.where(GRF_pred[:, :, 2] > threshold, GRF_pred[:, :, 2], torch.tensor(0.0))
    mask = GRF_pred[:,:,2] < threshold
    GRF_pred[mask] = 0

    CoP_pred = rotate_vis(CoP_pred)
    GRF_pred = rotate_vis(GRF_pred)
    drawpoint = rotate_vis(drawpoint)
    # print(drawpoint)
    pelvis = rotate_vis(pelvis_trans)
    vec = rotate_vis(vec)
    # print(CoP_pred)

    # print(drawpoint)


    # CoP_pred = torch.matmul(torch.tensor(Rx(-0.5*m.pi)), CoP_pred.transpose(-1, -2).type('torch.DoubleTensor')).transpose(-1, -2)
    # GRF_pred = torch.matmul(torch.tensor(Rx(-0.5*m.pi)), GRF_pred.transpose(-1, -2).type('torch.DoubleTensor')).transpose(-1, -2)

    arrow_renderables_somepoint = Arrows(
                drawpoint.numpy(),
                drawpoint.numpy() + vec.numpy(),
                color= (0 / 255, 191 / 255, 255 / 255, 1.0),
                is_selectable=True,
            )


    
    arrow_renderables_pred = Arrows(
                CoP_pred.numpy(),
                CoP_pred.numpy()+GRF_pred.numpy(),
                color= color_pred,
                is_selectable=True,
            )
    

    spheres = Spheres(pelvis.numpy(), color=(0 / 255, 191 / 255, 255 / 255, 1.0))

    # Display in the viewer.
    v = Viewer()
    v.run_animations = True
    v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    # v.scene.add(seq_amass, ptc_amass)
    v.scene.add(seq_amass)
    # v.scene.add(line_renderable)
    # v.scene.add(arrow_renderables)
    v.scene.add(arrow_renderables_pred)
    # v.scene.add(arrow_renderables_somepoint)
    # v.scene.add(spheres)
    # v.scene.add(line_renderable, arrow_renderables, arrow_renderable1)
    v.run()
