import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from show import *
from per_segment_anything import sam_model_registry, SamPredictor

import ipdb
import shutil
import json
import pandas as pd

output_root = 'draft_all'
show_others = True

class H_persam_seq (object):
    def __init__ (self, verbose=False):
        self.predictor = None
        self.prototype = None
        self.origin_mask = None
        self.target_embedding = None
        self.verbose = verbose

    def load_sam (self, sam_type='vit_t', sam_ckpt='weights/mobile_sam.pt', device='cuda'):
        print ("======> Load SAM") if self.verbose else None
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    def load_prototype (self, im, mask):
        print ("======> Load prototype") if self.verbose else None
        ref_mask = self.predictor.set_image (im, mask)
        ref_feat = self.predictor.features.squeeze().permute(1, 2, 0)
        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
        ref_mask = ref_mask.squeeze()[0]

        # Target feature extraction
        target_feat = ref_feat[ref_mask > 0]
        target_embedding = target_feat.mean(0).unsqueeze(0)
        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        target_embedding = target_embedding.unsqueeze(0)
        self.prototype = target_feat
        self.target_embedding = target_embedding

    def predict_once (self, im, mask=None, idx=None, best_only=True, persam=True):
        self.predictor.set_image (im)

        if mask is None:
            mask = self.origin_mask
        
        if persam:
            test_feat = self.predictor.features.squeeze ()

            # Cosine similarity
            C, h, w = test_feat.shape
            test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
            test_feat = test_feat.reshape(C, h * w)
            sim = self.prototype @ test_feat

            sim_img = sim.reshape(h, w, 1)
            sim_img = ((sim_img - sim_img.min ()) / (sim_img.max () - sim_img.min ()) * 255).cpu ().numpy ().astype (np.uint8)
            if show_others:
                sim_output_path = os.path.join(output_path, f'sim_{idx}.png')
                cv2.imwrite(sim_output_path, sim_img)

            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
            sim = self.predictor.model.postprocess_masks(sim, input_size=self.predictor.input_size, original_size=self.predictor.original_size).squeeze()

            # Positive-negative location prior
            topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
            topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
            topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

            # Obtain the target guidance for cross-attention layers
            sim = (sim - sim.mean()) / torch.std(sim)
            sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
            attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)
            target_embedding = self.target_embedding
        
        else:
            attn_sim = None
            target_embedding = None
            topk_xy = None
            topk_label = None

        # First-step prediction
        masks0, _, logits0, _ = self.predictor.predict(
            point_coords=topk_xy, 
            point_labels=topk_label, 
            multimask_output=True,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=target_embedding  # Target-semantic Prompting
        )
        
        best_iou = 0
        best_output = None
        
        ious_0 = [iou (masks0 [midx0], mask) for midx0 in range (masks0.shape [0])]
        bidx0 = np.array (ious_0).argmax ()

        
        # masks0 = avgpool (masks0.astype (np.float32))
        for midx0 in range (masks0.shape [0]):
            if best_only and (midx0 != bidx0):
                continue
            iou_0 = ious_0 [midx0]

            if show_others:
                print (midx0, iou_0)
                plt.figure(figsize=(8, 8))
                plt.imshow(im)
                show_mask(masks0[midx0], plt.gca())
                show_points(topk_xy, topk_label, plt.gca())
                plt.title(f"vis_{idx}_{midx0}_step0, iou:{iou_0}", fontsize=18)
                plt.axis('off')
                vis_mask_output_path = os.path.join(output_path, f'vis_{idx}_{midx0}_step0.png')
                with open(vis_mask_output_path, 'wb') as outfile:
                    plt.savefig(outfile, format='png')

                final_mask = masks0[midx0]
                mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
                mask_colors[final_mask, :] = np.array([[0, 0, 128]])
                mask_output_path = os.path.join(output_path, f'mask_{idx}_{midx0}_step0.png')
                cv2.imwrite(mask_output_path, mask_colors)

            # Cascaded Post-refinement-1

            #ipdb.set_trace ()
            masks1, _, logits1, _ = self.predictor.predict(
                        point_coords=topk_xy,
                        point_labels=topk_label,
                        mask_input=logits0[midx0: midx0 + 1, :, :], 
                        multimask_output=True)

            ious_1 = [iou (masks1 [midx1], mask) for midx1 in range (masks1.shape [0])]
            bidx1 = np.array (ious_1).argmax ()

            # masks1 = avgpool (masks1.astype (np.float32))
            for midx1 in range (masks1.shape [0]):
                if best_only and (midx1 != bidx1):
                    continue
                iou_1 = ious_1 [midx1]

                if show_others:
                    print (midx0, midx1, iou_1)
                    plt.figure(figsize=(8, 8))
                    plt.imshow(im)
                    show_mask(masks1[midx1], plt.gca())
                    show_points(topk_xy, topk_label, plt.gca())
                    plt.title(f"vis_{idx}_{midx0}_{midx1}_step1, iou:{iou_1}", fontsize=18)
                    plt.axis('off')
                    vis_mask_output_path = os.path.join(output_path, f'vis_{idx}_{midx0}_{midx1}_step1.png')
                    with open(vis_mask_output_path, 'wb') as outfile:
                        plt.savefig(outfile, format='png')

                    final_mask = masks1[midx1]
                    mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
                    mask_colors[final_mask, :] = np.array([[0, 0, 128]])
                    mask_output_path = os.path.join(output_path, f'mask_{idx}_{midx0}_{midx1}_step1.png')
                    cv2.imwrite(mask_output_path, mask_colors)

                # Cascaded Post-refinement-2
                y, x = np.nonzero(masks1[midx1])
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])
                masks2, _, _, _ = self.predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    box=input_box[None, :],
                    mask_input=logits1[midx1: midx1 + 1, :, :], 
                    multimask_output=True)

                ious_2 = [iou (masks2 [midx2], mask) for midx2 in range (masks2.shape [0])]
                bidx2 = np.array (ious_2).argmax ()
                # masks2 = masks2.astype (np.float32)
                for midx2 in range (masks2.shape [0]):
                    if best_only and (midx2 != bidx2):
                        continue
                    iou_2 = ious_2 [midx2]

                    final_mask = masks2[midx2]
                    if show_others:
                        plt.figure(figsize=(8, 8))
                        plt.imshow(im)
                        show_mask(masks2[midx2], plt.gca())
                        show_points(topk_xy, topk_label, plt.gca())
                        plt.title(f"vis_{idx}_{midx0}_{midx1}_{midx2}_step2, iou:{iou_2}", fontsize=18)
                        plt.axis('off')
                        vis_mask_output_path = os.path.join(output_path, f'vis_{idx}_{midx0}_{midx1}_{midx2}_step2.png')
                        with open(vis_mask_output_path, 'wb') as outfile:
                            plt.savefig(outfile, format='png')
                        
                        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
                        mask_colors[final_mask, :] = np.array([[0, 0, 128]])
                        mask_output_path = os.path.join(output_path, f'mask_{idx}_{midx0}_{midx1}_{midx2}_step2.png')
                        cv2.imwrite(mask_output_path, mask_colors)

                        concatenated_img = concatenate_images(output_path, idx, midx0, midx1, midx2)
                        concatenated_output_path = os.path.join(output_path, f'concatenated_{idx}_{midx0}_{midx1}_{midx2}_step2.png')
                        #print (concatenated_img.shape)
                        cv2.imwrite(concatenated_output_path, concatenated_img)
                        plt.close ('all')

                        print (midx0, midx1, midx2, iou_2)
                    if iou_2 > best_iou or best_only:
                        bidx0, bidx1, bidx2 = midx0, midx1, midx2
                        best_iou = iou_2
                        best_output = final_mask

        info = {'seq': f'{bidx0}_{bidx1}_{bidx2}', 'topk_xy':topk_xy, 'topk_label':topk_label}
        return best_output, info
    
    def fit_predict_once (self, im, mask, idx=None, best_only=True, persam=True):
        self.load_prototype (im, mask)
        return self.predict_once (im, mask, idx, best_only, persam)
    
    def fit_seq (self, ims, masks, output_path, best_only=True, refresh_mask=True, use_gt=True, persam=True):
        result = []
        it_ims = iter (ims)
        it_masks = iter (masks)
        prev_mask = None
        for idx, im in enumerate (it_ims):
            mask = next (it_masks)
            if idx == 0:
                pred, info = self.fit_predict_once (im, mask, idx, best_only, persam)
                pred_iou = iou (pred, mask)
                prev_mask = mask if use_gt else np.repeat (np.expand_dims(pred * 255, axis=-1), 3, axis=-1).astype (np.uint8)
            else:
                pred, info = self.predict_once (im, prev_mask, idx, best_only, persam)
                pred_iou = iou (pred, mask)
                prev_mask = mask if use_gt else np.repeat (np.expand_dims(pred * 255, axis=-1), 3, axis=-1).astype (np.uint8)
                if refresh_mask:
                    self.load_prototype (im, prev_mask)

            best_output_path = f'{output_path}/{idx:02d}_best_{info ["seq"]}_{pred_iou:.4f}.png'
            cv2.imwrite(best_output_path, (pred * 255).astype (np.uint8))
            bv_output_path = f'{output_path}/{idx:02d}_bv_{info ["seq"]}_{pred_iou:.4f}.png'
            save_im_with_mask (im, pred, bv_output_path, info ['topk_xy'], info ['topk_label'], f'{idx:02d}_bv_{info ["seq"]}_{pred_iou:.4f}')
            best_output_path = f'{output_path}/{idx:02d}_gt.png'
            cv2.imwrite(best_output_path, mask)
            bv_output_path = f'{output_path}/{idx:02d}_mv.png'
            save_im_with_mask (im, prepare_iou (mask), bv_output_path, None, None, f'{idx:02d}_mv')
            print (idx, pred_iou)
            result.append (pred_iou)
        return result

def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
        
    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label

def concatenate_images(root, test_idx, step0, step1, step2):
    # Load images
    vis_step0 = cv2.imread(f'{root}/vis_{test_idx}_{step0}_step0.png')
    vis_step1 = cv2.imread(f'{root}/vis_{test_idx}_{step0}_{step1}_step1.png')
    vis_step2 = cv2.imread(f'{root}/vis_{test_idx}_{step0}_{step1}_{step2}_step2.png')
    concatenated_vis = cv2.hconcat([vis_step0, vis_step1, vis_step2])
    mask_step0 = cv2.imread(f'{root}/mask_{test_idx}_{step0}_step0.png')
    mask_step1 = cv2.imread(f'{root}/mask_{test_idx}_{step0}_{step1}_step1.png')
    mask_step2 = cv2.imread(f'{root}/mask_{test_idx}_{step0}_{step1}_{step2}_step2.png')
    concatenated_mask = cv2.hconcat([mask_step0, mask_step1, mask_step2])

    height1, width1 = concatenated_vis.shape[:2]
    height2, width2 = concatenated_mask.shape[:2]

    if width1 < width2:
        new_height = width2 * height1 // width1
        img1_resized = cv2.resize(concatenated_vis, (width2, new_height))
        img2_resized = concatenated_mask
    else:
        new_height = width1 * height2 // width2
        img1_resized = cv2.resize(concatenated_vis, (width1, height1))
        img2_resized = cv2.resize(concatenated_mask, (width1, new_height))

    concatenated_img = cv2.vconcat([img1_resized, img2_resized])

    return concatenated_img    

def dice_coefficient (predicted, target):
    intersection = np.sum(predicted * target)
    union = np.sum(predicted ** 2) + np.sum(target ** 2)
    dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
    return dice

def prepare_iou (im):
    im = im.astype (np.float32)
    while len (im.shape) > 2:
        im = im.mean (np.argmin (im.shape))
    assert im.min () >= 0
    if im.max () > 1:
        im = im / im.max ()
    return im 

def iou (predicted, target):
    x1 = prepare_iou (predicted)
    x2 = prepare_iou (target)
    #print (predicted.shape, target.shape, predicted.max (), target.max (), predicted.min (), target.min ())
    dice_coef = dice_coefficient(x1, x2)
    iou_value = dice_coef / (2.0 - dice_coef)
    return iou_value

def load_img (path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def avgpool (x):
    return (x [:, 0::2, 0::2] + x [:, 0::2, 1::2] + x [:, 1::2, 0::2] + x [:, 1::2, 1::2]) / 4

def save_im_with_mask (im, mask, path, topk_xy=None, topk_label=None, title=''):
    plt.figure(figsize=(8, 8))
    plt.imshow(im)
    show_mask(mask, plt.gca())
    if (topk_xy is not None) and (topk_label is not None):
        show_points(topk_xy, topk_label, plt.gca())
    plt.title(title, fontsize=18)
    plt.axis('off')
    with open(path, 'wb') as outfile:
        plt.savefig(outfile, format='png')
    plt.close ('all')

if __name__ == '__main__':
    if os.path.exists (output_root):
        shutil.rmtree (output_root)
    os.makedirs (output_root)

    model = H_persam_seq ()
    model.load_sam ()

    images_path = './data/Images'
    masks_path = './data/Annotations'
    

    ims = [load_img (f'./data/Images/BD4RDH/{i:02d}.png') for i in range (20)]
    masks = [load_img (f'./data/Annotations/BD4RDH/{i:02d}.png') for i in range (20)]

    exp = {}

    cfgs = [dict (best_only=bo, persam=ps) for bo in [True, False] for ps in [True, False]]
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    for idx, cfg in enumerate (cfgs):
        name = '_'.join ([str (i) for i in cfg.values ()])
        output_path = f'./{output_root}/{name}'
        os.makedirs (output_path, exist_ok=True)
        with open (f'{output_path}/cfg.json', 'w', encoding='utf-8') as f:
            json.dump (cfg, f) 
        print (f'setting {idx}:')
        print (cfg)
        ious = model.fit_seq (ims, masks, output_path, **cfg)
        with open (f'{output_path}/ious.json', 'w', encoding='utf-8') as f:
            json.dump (cfg, f)
        plt.figure (figsize=(12, 12))
        plt.title ('global iou')
        plt.plot (range (len (ious)), ious, c=colors [idx], label=name)
        plt.legend ()
        plt.savefig (f'{output_path}/summary.png')
        plt.close ('all')
        exp [name] = ious
    
    
    plt.figure (figsize=(12, 12))
    plt.title ('global iou')
    for idx, (name, ious) in enumerate (exp.items ()):
        plt.plot (range (len (ious)), ious, c=colors [idx], label=name)
    plt.legend ()
    plt.savefig (f'{output_root}/global.png')
    plt.close ('all')
    pd.DataFrame (exp).to_csv (f'{output_root}/global.csv')
    

    

