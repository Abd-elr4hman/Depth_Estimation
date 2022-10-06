from numpy import float16
import torch
import cv2
import os
import io
import json

import tempfile

from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

def model_fn(model_dir):
    # get device
    device= get_device()

    model = DPTDepthModel(
        path=os.path.join(model_dir, "model.pth"),
        backbone="vitl16_384",
        non_negative=True,
    )

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
    
    count=0 
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.release()
            break

        # Preprocess image
        img_input = preprocess(frame)
        #print(img_input.shape)
    
        prediction = predict(img_input , model)
        #print(prediction.shape)

        # cast to float16
        prediction= prediction.astype(float16)

        # change to list (json serializable)
        prediction= prediction.tolist()
    
        # append prediction to predictions list
        predictions.append(prediction)
        count+=1
        print(count)

    return json.dumps(predictions)
    #return prediction


def predict(img_input, model):
    # get device
    device= get_device()

    # input image original size
    input_img_size= (384, 672)


    # predict
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        #print(sample.shape)
        
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=input_img_size,
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        #print(prediction.shape)

    return prediction


def preprocess(img):

    # array: RGB image (0-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    net_w, net_h = 384, 384
    resize_mode = "minimal"
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
        )
    img_input= transform({"image": img})["image"]

    return img_input

'''
def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img
'''