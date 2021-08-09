# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from DEKR.lib.dataset.transforms import FLIP_CONFIG
from DEKR.lib.utils.transforms import up_interpolate


def get_locations(output_h, output_w, device):
    shifts_x = torch.arange(
        0, output_w, step=1,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, output_h, step=1,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)

    return locations


def get_reg_poses(offset, num_joints):
    #modificações para tratamento de batch de poses
    b,_, h, w = offset.shape
    offset = offset.permute(0,2, 3, 1).reshape(b,h*w, num_joints, 2)
    locations = get_locations(h, w, offset.device)
    locations = locations[:, None, :].expand(-1, num_joints, -1)
    # poses =  - locations
    poses = locations - offset[:]

    return poses


def offset_to_pose(offset, flip=True, flip_index=None):
    
    batch_size,num_offset, h, w = offset.shape
    num_joints = int(num_offset/2)
    reg_poses = get_reg_poses(offset, num_joints)

    if flip:
        reg_poses = reg_poses[:,:, flip_index, :]
        reg_poses[:,:, :, 0] = w - reg_poses[:, :, :, 0] - 1

    reg_poses = reg_poses.contiguous().view(batch_size,h*w, 2*num_joints).permute(0,2,1)
    reg_poses = reg_poses.contiguous().view(batch_size,-1,h,w).contiguous()

    return reg_poses


def get_multi_stage_outputs(cfg, model, image, with_flip=False):
    # forward
    heatmap, offset = model(image)
    posemap = offset_to_pose(offset, flip=False)

    if with_flip:
        if 'coco' in cfg.DATASET.DATASET:
            flip_index_heat = FLIP_CONFIG['COCO_WITH_CENTER']
            flip_index_offset = FLIP_CONFIG['COCO']
        elif 'crowd_pose' in cfg.DATASET.DATASET:
            flip_index_heat = FLIP_CONFIG['CROWDPOSE_WITH_CENTER']
            flip_index_offset = FLIP_CONFIG['CROWDPOSE']
        else:
            raise ValueError('Please implement flip_index \
                for new dataset: %s.' % cfg.DATASET.DATASET)

        image = torch.flip(image, [3])
        image[:, :, :, :-3] = image[:, :, :, 3:]
        heatmap_flip, offset_flip = model(image)

        heatmap_flip = torch.flip(heatmap_flip, [3])
        heatmap = (heatmap + heatmap_flip[:, flip_index_heat, :, :])/2.0

        posemap_flip = offset_to_pose(offset_flip, flip_index=flip_index_offset)
        posemap = (posemap + torch.flip(posemap_flip, [3]))/2.0

    return heatmap, posemap


def hierarchical_pool(cfg, heatmap):
    pool1 = torch.nn.MaxPool2d(3, 1, 1)
    pool2 = torch.nn.MaxPool2d(5, 1, 2)
    pool3 = torch.nn.MaxPool2d(7, 1, 3)
    map_size = (heatmap.shape[-2]+heatmap.shape[-1])/2.0
    if map_size > cfg.TEST.POOL_THRESHOLD1:
        maxm = pool3(heatmap)
    elif map_size > cfg.TEST.POOL_THRESHOLD2:
        maxm = pool2(heatmap)
    else:
        maxm = pool1(heatmap)

    return maxm


def get_maximum_from_heatmap(cfg, heatmap):
    maxm = hierarchical_pool(cfg, heatmap)
    maxm = torch.eq(maxm, heatmap).float()
    heatmap = heatmap * maxm
    scores = heatmap.view(heatmap.shape[0],heatmap.shape[-1]*heatmap.shape[-2])
    scores, pos_ind = scores.topk(cfg.DATASET.MAX_NUM_PEOPLE,dim=1)
    pos_ind = pos_ind.cpu()
    # block_size = scores.shape[1]//heatmap.shape[0]
    # selecao = (scores > (cfg.TEST.KEYPOINT_THRESHOLD)).view(heatmap.shape[0]*cfg.DATASET.MAX_NUM_PEOPLE)
    selecao = [(s > cfg.TEST.KEYPOINT_THRESHOLD).nonzero().reshape(-1) for s in scores]
    # selecao = (scores > (cfg.TEST.KEYPOINT_THRESHOLD)).nonzero()[:,1].view(heatmap.shape[0]*cfg.DATASET.MAX_NUM_PEOPLE)
    # select_ind = torch.split(selecao.cpu(), cfg.DATASET.MAX_NUM_PEOPLE, dim=0)
    pos_ind = [s[ind] for s,ind in zip(pos_ind,selecao)]
    scores = [s[ind] for s,ind in zip(scores,selecao)]
    # scores = scores[select_ind][:, 0]
    
# [s[ind.reshape(-1)] for s,ind in zip(scores,pos_ind)]
    return pos_ind, scores

def aggregate_results(
        cfg, heatmap_sum, poses, heatmap, posemap, scale
):
    """
    Get initial pose proposals and aggregate the results of all scale.

    Args: 
        heatmap (Tensor): Heatmap at this scale (1, 1+num_joints, w, h)
        posemap (Tensor): Posemap at this scale (1, 2*num_joints, w, h)
        heatmap_sum (Tensor): Sum of the heatmaps (1, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """

    ratio = cfg.DATASET.INPUT_SIZE*1.0/cfg.DATASET.OUTPUT_SIZE
    reverse_scale = ratio/scale
    h, w = heatmap[0].size(-1), heatmap[0].size(-2)
    
    heatmap_sum += up_interpolate(
        heatmap,
        size=(int(reverse_scale*w), int(reverse_scale*h)),
        mode='bilinear'
    )
    # center_heatmap1 = heatmap[0,-1:]#heatmap[:,1,:]
    # center_heatmap2 = heatmap[1,-1:]
    # center_heatmap = heatmap[:,None,1]
    center_heatmap = heatmap[:,-1:]#heatmap[:,1,:]
    
    pose_ind, ctr_score = get_maximum_from_heatmap(cfg, center_heatmap)
    posemap = posemap.permute(0, 2, 3, 1).view(posemap.shape[0],h*w, -1, 2)
    pose = [reverse_scale*posemap[i][ind] for i,ind in enumerate(pose_ind)]
    ctr_score = [c[:,None].expand(-1,p.shape[-2])[:,:,None] for c,p in zip(ctr_score,pose)]
    for i, l in enumerate(poses):
        l.append(torch.cat([pose[i], ctr_score[i]], dim=2))
    # poses.append([torch.cat([p, c], dim=2) for p,c in zip(pose,ctr_score)])

    return heatmap_sum, poses
