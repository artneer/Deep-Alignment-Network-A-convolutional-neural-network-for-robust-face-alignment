import os
import sys
import glob
import shutil


data_dir = sys.argv[1]
data_type = sys.argv[2]

ptv_list = []
p = os.path.join(data_dir, '*.ptv')
ptv_list.extend(glob.glob(p))

save_path = os.path.join(data_dir, data_type)

try:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
except OSError:
    print('Error: Creating directory. ' + save_path)

for x in ptv_list:
    filename = os.path.basename(x)
    shutil.move(x, os.path.join(save_path, filename))
