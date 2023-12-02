import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import pandas as pd
import glob

from collections import OrderedDict

import torch
from scipy.signal import butter, lfilter
import math as m
import data_processing_utility as DataProcess

participants_gender = {
    "s001_20220513" : "female",
    "s002_20220520" : "male",
    "s003_20220523" : "female",
    "s004_20220524" : "male",
    "s005_20220610" : "male",
    "s006_20220614" : "female",
    "s007_20220705" : "female",
}

participants_ID = {
    "s001" : "s001_20220513",
    "s002" : "s002_20220520",
    "s003" : "s003_20220523",
    "s004" : "s004_20220524",
    "s005" : "s005_20220610",
    "s006" : "s006_20220614",
    "s007" : "s007_20220705",
}


motiontype = {
    'tree' : 'yoga',
    'treearms' : 'yoga',
    'chair' : 'yoga',
    'squat' : 'yoga',
    'worrier1' : 'warrior',
    'worrier2' : 'warrior',
    'sidestretch' : 'side_stretch',
    'dog' : 'hand',
    'jumpingjack' : 'jump',
    'walk' : 'walk',
    'walk_00': 'walk',
    'hopping' : 'hopping',
    'ballethighleg' : 'ballet_high',
    'balletsmalljump' : 'ballet_jump',
    'whirl' : 'dance',
    'lambadadance' : 'yoga',
    'taichi' : 'taichi',
    'step' : 'stairs',
    'tennisserve' : 'tennis',
    'tennisgroundstroke' : 'tennis',
    'soccerkick' : 'kicking',
    'idling' : 'idling',
    'idling_00' : 'idling',
    'static' : 'static',
    'ballet_high_leg' : 'ballet_high'
}

def get_trial_name(file, system):
    if system == 'Windows':
        return os.path.splitext(file)[0].split('\\')[-1]
    else:
        return os.path.splitext(file)[0].split('/')[-1]

def resample(left, right, rate, character):
    # left and right are csv
    if not os.path.exists(left) or not os.path.exists(right):
        # print("Left force data: " + left)
        # print("Right force data: " + right)
        return None
    else:
        data_left = pd.read_csv(left).multiply(0.1)

        data_right = pd.read_csv(right).multiply(0.1)

        copR=DataProcess.extractData(data_right, 'COP')
        grfR=DataProcess.extractData(data_right, 'Force')

        copL=DataProcess.extractData(data_left, 'COP')
        grfL=DataProcess.extractData(data_left, 'Force')

        flip_grfL = DataProcess.rotate_origin(grfL)
        flip_grfR = DataProcess.rotate_origin(grfR)
        
        contact_force = {}

        l_cop = copL[::rate]*0.01
        r_cop = copR[::rate]*0.01
        l_grf = flip_grfL[::rate]*0.01
        r_grf = flip_grfR[::rate]*0.01
        contact_force["CoP"] = torch.permute(torch.tensor([l_cop.tolist(),r_cop.tolist()]), (1,0,2))
        contact_force["GRF"] = torch.permute(torch.tensor([l_grf.tolist(),r_grf.tolist()]), (1,0,2))
        # smpl

        return contact_force

def resample_walk(index, left, right, third, rate, character):
    # left and right are csv
    if not os.path.exists(left) or not os.path.exists(right) or not os.path.exists(third):
        # print("Left force data: " + left)
        # print("Right force data: " + right)
        if(index == 0):
            print("Second left force data: " + third)
        else:
            print("Second right force data: " + third)
        return None
    else:
        if index == 0:
            df1 = pd.read_csv(left).multiply(0.1)
            df2 = pd.read_csv(third).multiply(0.1)
            half_len = len(df1) // 2
            data_left = pd.concat([df1[:half_len], df2[half_len:]])
            data_right = pd.read_csv(right).multiply(0.1)
        else:
            df1 = pd.read_csv(right).multiply(0.1)
            df2 = pd.read_csv(third).multiply(0.1)
            half_len = len(df1) // 2
            data_right = pd.concat([df1[:half_len], df2[half_len:]])
            data_left = pd.read_csv(left).multiply(0.1)
    
        copR=DataProcess.extractData(data_right, 'COP')
        grfR=DataProcess.extractData(data_right, 'Force')

        copL=DataProcess.extractData(data_left, 'COP')
        grfL=DataProcess.extractData(data_left, 'Force')


        flip_grfL = DataProcess.rotate_origin(grfL)
        flip_grfR = DataProcess.rotate_origin(grfR)
        
        contact_force = {}

        l_cop = copL[::rate]*0.01
        r_cop = copR[::rate]*0.01
        l_grf = flip_grfL[::rate]*0.01
        r_grf = flip_grfR[::rate]*0.01
        contact_force["CoP"] = torch.permute(torch.tensor([l_cop.tolist(),r_cop.tolist()]), (1,0,2))
        contact_force["GRF"] = torch.permute(torch.tensor([l_grf.tolist(),r_grf.tolist()]), (1,0,2))
        # print("running changed version")


        return contact_force


def assign_force_to_foot(trial, participant, rate, character, csvfolder, npyfolder, system):
    # print("Participant ID: " + participant)
    # print("Processing: " + trial)

    motion = trial[14:-2]

    if not os.path.exists(npyfolder):
        os.makedirs(npyfolder)

    save_npy_path = npyfolder + '/' + trial + '.npy'

    # for walking trials
    second_left_step_detected = False # Will be set to True if a second left step is detected for a walk trial
    second_right_step_detected = False # Will be set to True if a second right step is detected for a walk trial

    if motiontype[motion] == 'yoga' or motiontype[motion] == 'hopping':
        # left on 1 and right on 2
        left = 1
        right = 2
    elif motiontype[motion] == 'warrior':
        left = 1
        if participant == 's001_20220513' or participant == 's002_20220520' or participant == 's003_20220523' or participant == 's005_20220610' or participant == 's006_20220614':
            right = 2
        else:
            right = 3
    elif motiontype[motion] == 'jump':
        if participant == 's001_20220513' or participant == 's004_20220524':
            left = 0
            right = 0
            pass
        else:
            left = 1
            right = 2
            pass
    elif motiontype[motion] == 'taichi':
        if participant == 's004_20220524':
            second_left_step_detected = True
            left = 2
            left_2 = 1
            right = 3
        else:
            left = 1
            right = 2
    elif motiontype[motion] == 'side_stretch':
        # left on 1 and right on 3
        left = 1
        right = 3
    elif motiontype[motion] == 'hand':
        # dog motion
        # left on 1 and right on 4
        # left hand on 5 and right hand on 3
        left = 1
        right = 4
    elif motiontype[motion] == 'tennis':
        if participant == 's001_20220513':
            # s001: left on 2 and right on 1
            left = 2
            right = 1
        elif participant == 's007_20220705' or participant == 's006_20220614':
            # left on 1 and right on 2
            left = 1
            right = 2
        else:
            # Otherwise, skip as we're only interested in s001 and s007 for tennis motions
            left = 0
            right = 0
            pass
    elif motiontype[motion] == 'kicking':
        if participant == 's001_20220513':
            # s001: right on 1,2,3 and left on 4,5
            # print("s001 soccer kick, skip for now..")
            left = 5
            right = 3
            # pass
        else:
            # second_left_step_detected = True
            left = 4
            right = 2
            # left_2 = 4
    elif motiontype[motion] == 'walk':
        if participant == 's001_20220513':
            second_left_step_detected = True
            left = 1 #+3
            right = 2
            left_2 = 3

        elif participant == 's002_20220520':
            if trial[-1] == '1':
                second_right_step_detected = True
                left = 2
                right = 1 #+3
                right_2 = 3
            else:
                second_left_step_detected = True
                left = 1 #+3
                right = 2
                left_2 = 3

        elif participant == 's003_20220523':
            second_left_step_detected = True
            left = 1 #+3
            right = 2
            left_2 = 3

        elif participant == 's004_20220524':
            second_right_step_detected = True
            left = 2
            right = 1 #+3
            right_2 = 3

        elif participant == 's005_20220610':
            if trial[-1] == '1':
                second_right_step_detected = True
                left = 2
                right = 1 #+3
                right_2 = 3
            else:
                second_left_step_detected = True
                left = 1 #+3
                right = 2
                left_2 = 3

        elif participant == 's006_20220614':
            second_right_step_detected = True
            left = 2
            right = 1 #+3
            right_2 = 3
        elif participant == 's007_20220705':
            if trial[-1] == '4':
                second_right_step_detected = True
                left = 2
                right = 1 #+3
                right_2 = 3
            else:
                second_left_step_detected = True
                left = 1 #+3
                right = 2
                left_2 = 3
        else:
            pass
    elif motiontype[motion] == "idling":
        # if participant == 's001_20220513':
        #     left = 0
        #     right = 0
        #     pass
        # else:
            left = 1
            right = 2
    elif motiontype[motion] == "ballet_jump":
        if participant == 's001_20220513':
            left = 2
            right = 3
        elif participant == 's002_20220520' or participant == 's003_20220523' or participant == 's004_20220524':
            left = 1
            right = 2
        else:
            left = 1
            right = 4
    elif motiontype[motion] == "ballet_high":
        left = 1
        right = 3
    else:
        # print("Skipping Motions..")
        left = 0
        right = 0
        pass

    if system == "Windows":
        bar = '\\'
    else:
        bar = '/'

    forceFile_left = csvfolder + bar + trial + '_f_' + str(left) + '.csv'
    forceFile_right = csvfolder + bar + trial + '_f_' + str(right) + '.csv'

    if second_left_step_detected:
        forceFile_third = csvfolder + bar + trial + '_f_' + str(left_2) + '.csv'
        resampledforce = resample_walk(0, forceFile_left, forceFile_right, forceFile_third, rate, character)
    elif second_right_step_detected:
        forceFile_third = csvfolder + bar + trial + '_f_' + str(right_2) + '.csv'
        resampledforce = resample_walk(1, forceFile_left, forceFile_right, forceFile_third, rate, character)
    else:
        resampledforce = resample(forceFile_left, forceFile_right, rate, character)

    if not resampledforce:
        pass
        # print("No resampled force.. Motion was Skipped")
    else:
        if not os.path.exists(save_npy_path):
            np.save(save_npy_path, resampledforce)
            # print("processed: "+trial)
        else:
            pass
            # print("file exists. Skipping..")    

