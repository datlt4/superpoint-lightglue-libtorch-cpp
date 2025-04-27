# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger

import torch
from torch import nn
from typing import Tuple, List


@torch.jit.script
def simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    zeros = torch.zeros_like(scores)
    max_mask = scores == torch.nn.functional.max_pool2d(
        scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
    )
    for _ in range(2):
        supp_mask = torch.nn.functional.max_pool2d(
            max_mask.float(), kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        ) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == torch.nn.functional.max_pool2d(
            supp_scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


@torch.jit.script
def top_k_keypoints(keypoints: torch.Tensor, scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if k >= scores.shape[0]:
        return keypoints, scores
    top_scores, indices = torch.topk(scores, k, dim=0, largest=True, sorted=True)
    return keypoints[indices], top_scores


@torch.jit.script
def sample_descriptors(keypoints: torch.Tensor, descriptors: torch.Tensor, s: int) -> torch.Tensor:
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2.0 + 0.5
    size_tensor = torch.tensor([float(w * s - s / 2.0 - 0.5), float(h * s - s / 2.0 - 0.5)], device=keypoints.device)
    keypoints = keypoints / size_tensor
    keypoints = keypoints * 2.0 - 1.0  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=True
    )
    descriptors = torch.nn.functional.normalize(descriptors.view(b, c, -1), p=2.0, dim=1)
    return descriptors

class SuperPoint(torch.nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """

    def __init__(self):
        super().__init__()  # Update with default configuration.
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0) # 256 is the default descriptor dimension

        
        url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"  # noqa
        self.load_state_dict(torch.hub.load_state_dict_from_url(url))

    def forward(self, image: torch.Tensor) -> tuple:
        # image: [1, 1, h, w] - grayscale image in torch.float32 format
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        # scores = scores.unsqueeze(1)
        scores = simple_nms(scores, 4) # nms_radius
        scores = scores.squeeze(1)

        # Discard keypoints near the image borders
        pad = 4 # remove borders
        scores = scores.clone()
        scores[:, :pad, :] = -1
        scores[:, :, :pad] = -1
        scores[:, -pad:, :] = -1
        scores[:, :, -pad:] = -1

        # Extract keypoints
        mask = scores[0] > 0.0005
        ys, xs = torch.where(mask)
        score_vals = scores[0][ys, xs]
        kpts = torch.stack((xs, ys), dim=-1).float()

        # Top K keypoints
        if kpts.shape[0] > 512:
            topk = torch.topk(score_vals, 512, dim=0, largest=True, sorted=True)
            kpts = kpts[topk.indices]
            score_vals = topk.values

        # Compute dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2.0, dim=1)

        # Sample descriptors
        desc = sample_descriptors(kpts[None], descriptors, 8)[0]

        return kpts[None], score_vals[None], desc[None].transpose(-1, -2).contiguous()
        {
            "keypoints": torch.stack(keypoints, 0),
            "keypoint_scores": torch.stack(scores, 0),
            "descriptors": torch.stack(descriptors, 0).transpose(-1, -2).contiguous(),
        }
