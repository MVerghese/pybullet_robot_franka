import numpy as np 
import pandas as pd 
import os 
import sys 
import time
from collections import OrderedDict
from PIL import Image 
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import pickle
from tqdm import tqdm 
import random
from einops import rearrange
sys.path.append('/home/yiqiw2/experiment/RoboNLP/LaViLa')


from lavila.data.video_transforms import Permute
from lavila.data.datasets import get_frame_ids, video_loader_by_frames
from lavila.models.models import VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL
from lavila.models import models
from lavila.models.tokenizer import MyGPT2Tokenizer
from eval_narrator import decode_one
from lavila.models.utils import inflate_positional_embeds
from lavila.utils.preprocess import  generate_tokenizer

# ckpt_path = '/home/yiqiw2/experiment/RoboNLP/lavila_ckpt/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth'
# device = 'cuda:1'

def read_video_frames(frames, frame_num = 16):

    index = np.linspace( 0, len(frames), frame_num, endpoint = False, dtype = int)
    selected_frames = []
    for f_idx in index:
        frame = frames[f_idx]
        frame=rearrange(frame, 'H W C -> 1 C H W')
        assert len( frame.shape ) == 4
        selected_frames.append(frame)
    selected_frames= np.concatenate(selected_frames, axis = 0)

    return selected_frames

class LaViLa_Interface:
    def __init__(self, ckpt_path, cuda_device=0, dual_cuda_device=None, clip_length=4):

        if dual_cuda_device is None:
            dual_cuda_device = cuda_device
        self.cuda_device = cuda_device

        self.dual_cuda_device = dual_cuda_device

        print('Dual cuda device: {}'.format(dual_cuda_device))
        self.ckpt_path = ckpt_path
        self.clip_length =  clip_length
        self.load_dual_model(self.ckpt_path, dual_cuda_device, clip_length)
    
    def load_dual_model(self, ckpt_path, cuda_device, clip_length = 4):
        torch.cuda.set_device(cuda_device)
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():

            state_dict[k.replace('module.', '')] = v
        
        old_args = ckpt['args']

        print('=> creating model: {}'.format(old_args.model))

        #CL Args
        # clip_length = 4

        self.dual_model = getattr(models, old_args.model)(
            # text_use_cls_token=old_args.use_cls_token,
            # project_embed_dim=old_args.project_embed_dim,
            gated_xattn=False if 'gated_xattn' not in old_args else old_args.gated_xattn,
            timesformer_gated_xattn=False if 'timesformer_gated_xattn' not in old_args else old_args.timesformer_gated_xattn,
            timesformer_freeze_space=False if 'timesformer_freeze_space' not in old_args else old_args.timesformer_freeze_space,
            freeze_lm_vclm=False if 'freeze_lm_vclm' not in old_args else old_args.freeze_lm_vclm,
            freeze_visual_vclm=False if 'freeze_visual_vclm' not in old_args else old_args.freeze_visual_vclm,
            num_frames=clip_length,
            drop_path_rate=0,
        )
        self.dual_model.cuda(device=cuda_device)
        if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
            # inflate weight
            print('=> inflating PE in models due to different frame numbers')
            state_dict = inflate_positional_embeds(
                self.dual_model.state_dict(), state_dict,
                num_frames=clip_length,
                load_temporal_fix='bilinear',
            )
        self.dual_model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}' (epoch {}, best_metric = {})".format(ckpt_path, ckpt['epoch'], ckpt['best_acc1']))
        self.dual_tokenizer = generate_tokenizer(old_args.model)
        crop_size = 224 if '336PX' not in old_args.model else 336
        self.dual_val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if ('OPENAI' not in old_args.model) else
             transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
        ])

    def eval_dual(self,frames,texts):
        # move frames to dual_cuda_device
        frames = frames.to(device=self.dual_cuda_device)
        texts = self.dual_tokenizer(texts)

        # print(texts.shape)
        # print(frames.shape)
        # Transpose frames to (T,H,W,C)
        frames = frames.permute(0,2,3,1)
        # print(frames.shape)
        frames = self.dual_val_transform(frames)
        frames = frames.unsqueeze(0)
        
        with torch.no_grad():
            texts = texts.cuda(non_blocking=True)
            frames = frames.cuda(non_blocking=True)
            text_embeddings = self.dual_model.encode_text(texts)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            image_features = self.dual_model.encode_image(frames)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # print("text_embeddings.shape: {}".format(text_embeddings.shape))
            # print("image_features.shape: {}".format(image_features.shape))
        return text_embeddings, image_features
    
    def video_feature(self, frames):
        frames = frames.to(device=self.dual_cuda_device)
        frames = frames.permute(0,2,3,1)
        # print(frames.shape)
        frames = self.dual_val_transform(frames)
        frames = frames.unsqueeze(0)
        
        with torch.no_grad():

            frames = frames.cuda(non_blocking=True)
            image_features = self.dual_model.encode_image(frames)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features
