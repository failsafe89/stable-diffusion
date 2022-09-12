#!/usr/bin/env python3

import socket
import struct
import atexit
import requests
import json

from concurrent.futures import ThreadPoolExecutor

from binascii import hexlify
from Crypto.Cipher import AES

import io
import cv2
import torch
import base64
import numpy as np
from PIL import Image
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

PORT = 5424
with open('register_passphrase.txt', 'r') as f:
    REGISTER_PASSPHRASE = f.read().strip()
with open('secret_key.txt', 'rb') as f:
    SECRET_KEY = f.read().strip()

SOCKET : socket.socket = None
CONNECTION : socket.socket = None

# message types
HANDSHAKE_REQUEST = 0
DREAM_PROMPT_REQUEST = 1
DREAM_IMAGE_REQUEST = 2

class CONNECTION_TERMINATED(Exception):
    pass

### Stable diffusion code ###
CONFIG_PATH = "configs/stable-diffusion/v1-inference.yaml"
MODEL_PATH = "models/ldm/stable-diffusion-v1/model.ckpt"
MODEL = None
SAMPLER = None
WM_ENCODER = None

# Options
BATCH_SIZE = 2
PRECISION_SCOPE = autocast
N_ITER = 2
SCALE = 7.5
C = 4 # latent channels
H = 512 # image height
W = 512 # image width
F = 8 # downsampling factor
DDIM_STEPS = 50
DDIM_ETA = 0.0
SEED = 42

def savePNG(img : Image) -> bytes:
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        return output.getvalue()

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def setup_sd_runtime():
    global MODEL
    global SAMPLER
    global WM_ENCODER
    global SEED
    # seed
    seed_everything(SEED)

    # load model
    config = OmegaConf.load(CONFIG_PATH)
    MODEL = load_model_from_config(config, MODEL_PATH)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MODEL = MODEL.to(device)

    # setup sampler
    SAMPLER = PLMSSampler(MODEL)
    #SAMPLER = DDIMSampler(model)

    # setup watermark
    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    WM_ENCODER = WatermarkEncoder()
    WM_ENCODER.set_watermark('bytes', wm.encode('utf-8'))

def diffuse(prompt, options):
    global MODEL
    global SAMPLER
    global WM_ENCODER
    global N_ITER
    global SCALE
    global BATCH_SIZE
    global C
    global H
    global W
    global F
    global DDIM_STEPS
    global DDIM_ETA

    batch = BATCH_SIZE if 'batchSize' not in options else options['batchSize']
    n_iter = N_ITER if 'nIter' not in options else options['nIter']
    width = W if 'width' not in options else options['width']
    height = H if 'height' not in options else options['height']
    ddim_steps = DDIM_STEPS if 'steps' not in options else options['steps']

    data = [batch * [prompt]]

    results = []

    with torch.no_grad():
        with PRECISION_SCOPE("cuda"):
            with MODEL.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if SCALE != 1.0:
                            uc = MODEL.get_learned_conditioning(batch * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = MODEL.get_learned_conditioning(prompts)
                        shape = [C, height // F, width // F]
                        samples_ddim, _ = SAMPLER.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=batch,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=SCALE,
                                                         unconditional_conditioning=uc,
                                                         eta=DDIM_ETA,
                                                         x_T=None)

                        x_samples_ddim = MODEL.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim #, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img = put_watermark(img, WM_ENCODER)
                            results.append(base64.encodebytes(savePNG(img)).decode('utf-8'))

                toc = time.time()

    print(f"Your samples are ready\n"
          f" \nEnjoy.")
    return results

### Stable Dreamer Server code ###

def padAES(data : bytes) -> bytes:
    paddingLength = 16 - (len(data) % 16)
    paddingChar = struct.pack('<B', paddingLength)
    return data + (paddingChar * paddingLength)

def stripAES(paddedData : bytes) -> bytes:
    paddingLength = struct.unpack('<B', paddedData[-1:])[0]
    if paddingLength > 16:
        raise Exception("Invalid padding")
    return paddedData[:-paddingLength]

def encrypt(data : bytes) -> bytes:
    paddedData = padAES(data)
    aes = AES.new(SECRET_KEY, AES.MODE_ECB)
    return aes.encrypt(paddedData)

def decrypt(data : bytes) -> bytes:
    aes = AES.new(SECRET_KEY, AES.MODE_ECB)
    return stripAES(aes.decrypt(data))
    

def shutdown():
    global SOCKET
    global CONNECTION

    print("shutting down cleanly")

    if SOCKET is not None:
        # SOCKET.shutdown()
        SOCKET.close()
        SOCKET = None

def registerAsDrearmer():
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(requests.post,
            'https://stabledream.app/registerDreamer', 
            json={'passphrase': REGISTER_PASSPHRASE, 'port': PORT},
            headers={'Content-Type': 'application/json', 'Origin': 'dreamer'}
        )

        doConnect()

        r = future.result(timeout=5)
        
        if not r.status_code == 200 or r.json()['dreamerId'] == 0:
            raise Exception(f'registerDreamer failed with status {r.status_code}')
        print(f"Dreamer registered [dreamerId: {r.json()['dreamerId']}]")
    return r.json()['dreamerId']

def waitForConnection():
    global SOCKET
    global CONNECTION
    SOCKET.listen(1)
    CONNECTION, _ = SOCKET.accept()

def doConnect():
    waitForConnection()
    print("Got connection")
    sendResponse(handleRequest(waitForRequest()))
    return True

def readbytes(num):
    global CONNECTION
    if CONNECTION is None:
        raise Exception("waitForRequest [ERROR]: CONNECTION is None")

    b = CONNECTION.recv(num)

    if len(b) == 0:
        # CONNECTION.shutdown()
        CONNECTION.close()
        CONNECTION = None
        raise CONNECTION_TERMINATED()

    return b

def waitForRequest():
    print("waiting for request len")
    requestLen = struct.unpack('<I', readbytes(4))[0]
    print(requestLen)
    print("waiting for request body")
    encdata = b''
    while len(encdata) < requestLen:
        encdata += readbytes(requestLen - len(encdata))
    print(f"read: {encdata}")
    request = decrypt(encdata)
    print(f"decrypted {request}")
    print(request)

    # request = {}
    return request

def handle_handshakeRequest(request):
    j = json.loads(request)
    return {'response': j['challenge']}

def handle_dreamPromptRequest(request):
    j = json.loads(request)
    results = diffuse(j['prompt'], j['options'])
    return {'prompt': j['prompt'], 'results': results}

def handle_dreamImageRequest(request):
    return request

def handleRequest(request : bytes):
    print("handling request")
    rtype = request[0]
    rbody = request[1:].decode('utf-8')
    print(f"request type: {rtype}")

    if rtype == HANDSHAKE_REQUEST:
        response = handle_handshakeRequest(rbody)
    elif rtype == DREAM_PROMPT_REQUEST:
        response = handle_dreamPromptRequest(rbody)
    elif rtype == DREAM_IMAGE_REQUEST:
        response = handle_dreamImageRequest(rbody)
    else:
        response = {'hello': 'world'}
        # response = {}
        print(request)

    return response

def sendResponse(response):
    global CONNECTION
    print("Sending response")
    if CONNECTION is None:
        raise Exception("sendResponse [ERROR]: CONNECTION is None")

    enc_response = encrypt(json.dumps(response).encode('utf-8'))
    print(hexlify(enc_response))
    print(len(enc_response))
    CONNECTION.send(struct.pack('<I', len(enc_response)))
    CONNECTION.send(enc_response)

if __name__ == '__main__':
    SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    SOCKET.bind(('0.0.0.0', PORT))

    atexit.register(shutdown)

    setup_sd_runtime()
    
    while True:
        if CONNECTION is None:
            registerAsDrearmer()
        try:
            sendResponse(handleRequest(waitForRequest()))
        except CONNECTION_TERMINATED as ct:
            print("Connection terminated")
            pass