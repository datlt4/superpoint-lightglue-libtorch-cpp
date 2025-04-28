# %% [markdown]
# ## Common

# %%
# pip3 install onnxruntime
import lightglue
import torch
import cv2
from time import time
# import numpy as np
# import onnxruntime as ort
# import onnx

# pip3 install tensorrt
# import tensorrt as trt
# from PIL import Image
# import pycuda.driver as cuda
# import pycuda.autoinit  # Initializes the first available GPU and sets up device context
# print(trt.__version__)

# providers=["CUDAExecutionProvider", "CPUExecutionProvider"]

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
print(device)

# %%
image0_path = "assets/image0.png"
size0 = 512
image1_path = "assets/image1.png"
size1 = 512

# %%
# Convert keypoints from PyTorch to OpenCV format
def convert_to_cv_keypoints(keypoints: torch.Tensor, scale):
    cv_keypoints = []
    for kp in keypoints:
        cv_keypoints.append(cv2.KeyPoint(kp[0].item()/scale[0], kp[1].item() / scale[1], 1))  # (x, y, size)
    return cv_keypoints

# Convert matches from PyTorch to OpenCV format
def convert_to_cv_matches(matches):
    cv_matches = []
    for match in matches:
        cv_matches.append(cv2.DMatch(match[0].item(), match[1].item(), 0))  # (queryIdx, trainIdx, distance)
    return cv_matches

# %% [markdown]
# ## SuperPoint - LightGlue : Pytorch Demo

# %%
extractor_pytorch = lightglue.SuperPoint().eval().to(device)  # load the extractor
matcher_pytorch = lightglue.LightGlue(features="superpoint").eval().to(device)

# %%
def pytorch_infer(image0_path: str, image1_path: str, resize0: int = 512, resize1: int = 512) -> None:
    start = time()
    image0, scale0 = lightglue.utils.load_image(image0_path, resize=resize0)
    image1, scale1 = lightglue.utils.load_image(image1_path, resize=resize1)

    keypoints0, _, descriptors0 = extractor_pytorch(image0.to(device))
    keypoints1, _, descriptors1 = extractor_pytorch(image1.to(device))
    kn0 = resize0 / 2.0 # keypoint normalize
    kn1 = resize1 / 2.0 # keypoint normalize
    matches01, mscore01 = matcher_pytorch(((keypoints0 - kn0) / kn0), descriptors0, ((keypoints1 - kn1) / kn1), descriptors1)
    print(type(matches01), type(mscore01))

    # Draw matches on the images and save the result image
    cv_kpts0 = convert_to_cv_keypoints(keypoints0[0], scale0)
    cv_kpts1 = convert_to_cv_keypoints(keypoints1[0], scale1)
    cv_matches = convert_to_cv_matches(matches01)
    match_image = cv2.drawMatches(cv2.imread(image0_path), cv_kpts0, cv2.imread(image1_path), cv_kpts1, cv_matches, None)
    end = time()
    print(f"Image0: {image0.shape}, Image1: {image1.shape}, Kpt0: {len(cv_kpts0)}, Kpt1: {len(cv_kpts1)}, Matches: {len(cv_matches)}, Inference time: {end - start:.2f} seconds")

    cv2.putText(img=match_image, text=f'Number of matches: {len(cv_matches)}', 
                org=(10, 30), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=3,
                lineType=4)
    cv2.imwrite(f"match_result_image.png", match_image)

# %%
pytorch_infer(image0_path, image1_path, size0, size1)

# %% [markdown]
# ## Convert the PyTorch Model to TorchScript

# %%
extractor_pytorch.eval()
matcher_pytorch.eval()

scripted_extractor = torch.jit.script(extractor_pytorch)
scripted_extractor.save("weights/superpoint_scripted.pt")

scripted_matcher = torch.jit.script(matcher_pytorch)
scripted_matcher.save("weights/lightglue_scripted.pt")


