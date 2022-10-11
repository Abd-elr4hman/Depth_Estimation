from models import UnetAdaptiveBins
import model_io
import torch
from torchvision import transforms
import torch.nn as nn
import io
import tempfile
import numpy as np
import json
import cv2
import os

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

def model_fn(model_dir):

    MIN_DEPTH = 1e-3
    MAX_DEPTH_KITTI = 80
    N_BINS = 256 
    
    # get device
    device= get_device()
    
    # build Net
    model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)

    # load pretrained wights
    pretrained_path = os.path.join(model_dir, "model.pth")
    model, _, _ = model_io.load_checkpoint(pretrained_path, model)

    model.to(device).eval()

    return model



def transform_fn(model, request_body, content_type, accept):
    
    device = get_device()

    # read image from io byte stream
    f = io.BytesIO(request_body)

    # create a temporary file and save the io byte image in it
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())

    # read video
    cap = cv2.VideoCapture(tfile.name)

    predictions=[]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.release()
            break

        # Preprocess image
        img_input = preprocess(frame)
        #print(img_input.shape)
        
        bin_centers, pred = predict(img_input , model)

        # absolute depth in meters to mm
        pred = pred*1000
        pred= pred.astype(np.uint16)

        pred= pred.tolist()
        predictions.append(pred)


    return json.dumps(predictions)
    #return pred



def predict(img_input, model):

    device = get_device()

    min_depth = 1e-3
    max_depth = 80

    saving_factor = 256

    # predict
    with torch.no_grad():
        bins, pred = model(img_input)
        pred = np.clip(pred.cpu().numpy(), min_depth, max_depth)

        # Flip
        img_input = torch.Tensor(np.array(img_input.cpu().numpy())[..., ::-1].copy()).to(device)
        pred_lr = model(img_input)[-1]
        pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], min_depth, max_depth)

        # Take average of original and mirror
        final = 0.5 * (pred + pred_lr)
        final = nn.functional.interpolate(torch.Tensor(final), img_input.shape[-2:],
                                          mode='bilinear', align_corners=True).cpu().squeeze().numpy()

        final[final < min_depth] = min_depth
        final[final > max_depth] = max_depth
        final[np.isinf(final)] = max_depth
        final[np.isnan(final)] = min_depth

        #final = (final * saving_factor).astype('uint16')

        centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        centers = centers.cpu().squeeze().numpy()
        centers = centers[centers > min_depth]
        centers = centers[centers < max_depth]

    return centers, final


def preprocess(input_frame):

    device = get_device()

    # read
    cv2_img = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

    # resize
    cv2_img = cv2.resize(cv2_img, (640, 480))

    # normalise
    np_img = cv2_img / 255.

    # to tensor
    np_img = torch.from_numpy(np_img.transpose((2, 0, 1)))
    normalize= transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    np_img= normalize(np_img).unsqueeze(0).float().to(device)
     
    return np_img

