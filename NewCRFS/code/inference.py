import torch
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable

import io
import tempfile
import numpy as np
import json
import cv2
import os

from newcrfs.networks.NewCRFDepth import NewCRFDepth
from newcrfs.utils import post_process_depth, flip_lr

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

def model_fn(model_dir):
 
    # get device
    device= get_device()
    
    # build Net
    model = NewCRFDepth(version='large07', inv_depth=False, max_depth=80)
    model = torch.nn.DataParallel(model)

    # load pretrained wights
    pretrained_path= os.path.join(model_dir, "model.pth")
    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint['model'])

    model.to(device).eval()

    return model



def transform_fn(model, request_body, content_type, accept):
    
    device = get_device()

    # read image from io byte stream
    f = io.BytesIO(request_body)

    # create a temporary file and save the io byte image in it
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())

    # Preprocess image
    batched_torch_input = preprocess(tfile.name)
    print(batched_torch_input.shape)
    
    pred = predict(batched_torch_input , model)

    # absolute depth in meters to mm
    pred = pred*1000
    pred= pred.astype(np.uint32)
    print(pred.shape)

    pred= pred.tolist()

    return json.dumps(pred)
    #return pred



def predict(batched_torch_input, model):

    device = get_device()

    # predict
    with torch.no_grad():
        batched_torch_input = Variable(batched_torch_input)
        print(batched_torch_input.shape)
        depth_est = model(batched_torch_input)
        
        post_process = True
        if post_process:
            batched_torch_input_flipped = flip_lr(batched_torch_input)
            depth_est_flipped = model(batched_torch_input_flipped)
            depth_est = post_process_depth(depth_est, depth_est_flipped)

        pred_depth = depth_est.cpu().numpy().squeeze()

    return pred_depth


def preprocess(path):

    device = get_device()

    # read
    image= cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #  normalize
    image = np.asarray(image, dtype=np.float32) / 255.0

    # resize
    image= cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)


    # to tensor
    torch_img = torch.from_numpy(image.transpose((2, 0, 1)))

    normalize= transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_img= normalize(torch_img).unsqueeze(0).to(device)
     
    return torch_img

