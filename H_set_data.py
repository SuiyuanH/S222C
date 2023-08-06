import os
import shutil
import cv2
import numpy as np

#start_idx = 55
#num_samples = 999
start_idx = 120
num_samples = 20
step = 1
mission = 'V'

if os.path.exists ('data'):
    shutil.rmtree ('data')
os.makedirs ('data/Annotations/BD4RDH')
os.makedirs ('data/Images/BD4RDH')

samples = []
old_idx = start_idx
new_idx = 0
while new_idx < num_samples:
    im = 'raw_data/imgs/BD4RDH_{}_raw.png'.format (old_idx)
    mask = 'raw_data/imgs/BD4RDH_{}_{}.png'.format (old_idx, mission)
    if not os.path.exists (im):
        print ('Warning: got less than expected!')
        break
    shutil.copyfile (im, 'data/Images/BD4RDH/{:02d}.png'.format (new_idx))
    
    samples.append (old_idx)
    if os.path.exists (mask):
        shutil.copyfile (mask, 'data/Annotations/BD4RDH/{:02d}.png'.format (new_idx))
    else:
        new_mask = np.zeros ([512, 512], dtype=np.uint8)
        cv2.imwrite ('data/Annotations/BD4RDH/{:02d}.png'.format (new_idx), new_mask)
    new_idx += 1
    old_idx += step

print ('Finish setting and got', samples)