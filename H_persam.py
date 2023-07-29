import numpy as np
import torch
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



def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='persam')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    
    args = parser.parse_args()
    return args


def main():

    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Annotations/'
    output_path = './outputs/' + args.outdir

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    
    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            persam(args, obj_name, images_path, masks_path, output_path)


def persam(args, obj_name, images_path, masks_path, output_path):

    print("\n------------> Segment " + obj_name)
    
    # Path preparation
    ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.png')
    ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
    test_images_path = os.path.join(images_path, obj_name)

    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
    

    print("======> Load SAM" )
    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()

    predictor = SamPredictor(sam)

    print("======> Obtain Location Prior" )
    # Image features encoding
    ref_mask = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]

    # Target feature extraction
    target_feat = ref_feat[ref_mask > 0]
    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)

    print('======> Start Testing')
    for test_idx in tqdm(range(len(os.listdir(test_images_path)))):
    
        # Load test image
        test_idx = '%02d' % test_idx
        test_image_path = test_images_path + '/' + test_idx + '.png'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        test_mask_path = os.path.join(masks_path, obj_name, test_idx + '.png')
        test_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE).astype (np.float32)
        test_mask = test_mask / test_mask.max ()

        # Image feature encoding
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()

        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = target_feat @ test_feat

        sim_img = sim.reshape(h, w, 1)
        sim_img = ((sim_img - sim_img.min ()) / (sim_img.max () - sim_img.min ()) * 255).cpu ().numpy ().astype (np.uint8)
        sim_output_path = os.path.join(output_path, f'sim_{test_idx}.png')
        cv2.imwrite(sim_output_path, sim_img)
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze()

        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

        # First-step prediction
        masks0, _, logits0, _ = predictor.predict(
            point_coords=topk_xy, 
            point_labels=topk_label, 
            multimask_output=True,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=target_embedding  # Target-semantic Prompting
        )
        assert masks0.max () <= 1 and masks0.min () >= 0
        best_iou = 0
        best_output = None
        for midx0 in range (masks0.shape [0]):
            
            iou_0 = iou (masks0 [midx0], test_mask)
            plt.figure(figsize=(8, 8))
            plt.imshow(test_image)
            show_mask(masks0[midx0], plt.gca())
            show_points(topk_xy, topk_label, plt.gca())
            plt.title(f"vis_{test_idx}_{midx0}_step0, iou:{iou_0}", fontsize=18)
            plt.axis('off')
            vis_mask_output_path = os.path.join(output_path, f'vis_{test_idx}_{midx0}_step0.png')
            with open(vis_mask_output_path, 'wb') as outfile:
                plt.savefig(outfile, format='png')

            final_mask = masks0[midx0]
            mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
            mask_colors[final_mask, :] = np.array([[0, 0, 128]])
            mask_output_path = os.path.join(output_path, f'mask_{test_idx}_{midx0}_step0.png')
            cv2.imwrite(mask_output_path, mask_colors)

            # Cascaded Post-refinement-1
            masks1, _, logits1, _ = predictor.predict(
                        point_coords=topk_xy,
                        point_labels=topk_label,
                        mask_input=logits0[midx0: midx0 + 1, :, :], 
                        multimask_output=True)

            for midx1 in range (masks1.shape [0]):
                iou_1 = iou (masks1 [midx1], test_mask)
                plt.figure(figsize=(8, 8))
                plt.imshow(test_image)
                show_mask(masks1[midx1], plt.gca())
                show_points(topk_xy, topk_label, plt.gca())
                plt.title(f"vis_{test_idx}_{midx0}_{midx1}_step1, iou:{iou_1}", fontsize=18)
                plt.axis('off')
                vis_mask_output_path = os.path.join(output_path, f'vis_{test_idx}_{midx0}_{midx1}_step1.png')
                with open(vis_mask_output_path, 'wb') as outfile:
                    plt.savefig(outfile, format='png')

                final_mask = masks1[midx1]
                mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
                mask_colors[final_mask, :] = np.array([[0, 0, 128]])
                mask_output_path = os.path.join(output_path, f'mask_{test_idx}_{midx0}_{midx1}_step1.png')
                cv2.imwrite(mask_output_path, mask_colors)

                # Cascaded Post-refinement-2
                y, x = np.nonzero(masks1[midx1])
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])
                masks2, _, _, _ = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    box=input_box[None, :],
                    mask_input=logits1[midx1: midx1 + 1, :, :], 
                    multimask_output=True)

                for midx2 in range (masks2.shape [0]):
                    iou_2 = iou (masks2 [midx2], test_mask)
                    plt.figure(figsize=(8, 8))
                    plt.imshow(test_image)
                    show_mask(masks2[midx2], plt.gca())
                    show_points(topk_xy, topk_label, plt.gca())
                    plt.title(f"vis_{test_idx}_{midx0}_{midx1}_{midx2}_step2, iou:{iou_2}", fontsize=18)
                    plt.axis('off')
                    vis_mask_output_path = os.path.join(output_path, f'vis_{test_idx}_{midx0}_{midx1}_{midx2}_step2.png')
                    with open(vis_mask_output_path, 'wb') as outfile:
                        plt.savefig(outfile, format='png')

                    final_mask = masks2[midx2]
                    mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
                    mask_colors[final_mask, :] = np.array([[0, 0, 128]])
                    mask_output_path = os.path.join(output_path, f'mask_{test_idx}_{midx0}_{midx1}_{midx2}_step2.png')
                    cv2.imwrite(mask_output_path, mask_colors)

                    concatenated_img = concatenate_images(output_path, test_idx, midx0, midx1, midx2)
                    concatenated_output_path = os.path.join(output_path, f'concatenated_{test_idx}_{midx0}_{midx1}_{midx2}_step2.png')
                    #print (concatenated_img.shape)
                    cv2.imwrite(concatenated_output_path, concatenated_img)
                    plt.close ('all')

                    if float (iou_2) > best_iou:
                        best_iou = float (iou_2)
                        best_output = concatenated_img

        best_output_path = os.path.join(output_path, f'best_{test_idx}.png')
        cv2.imwrite(best_output_path, best_output)


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

def iou (predicted, target):
    dice_coef = dice_coefficient(predicted, target)
    iou_value = dice_coef / (2.0 - dice_coef)
    iou_pr = '{:.4f}'.format (iou_value)
    return iou_pr

if __name__ == "__main__":
    main()
