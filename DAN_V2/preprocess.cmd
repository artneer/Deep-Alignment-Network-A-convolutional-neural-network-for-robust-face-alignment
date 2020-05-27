python preprocessing.py --input_dir=./data/train --output_dir=./prep/train --istrain=True --repeat=10 --img_size=112 --mirror_file=./Mirror68.txt
python preprocessing.py --input_dir=./data/valid --output_dir=./prep/valid --istrain=False --img_size=112
python preprocessing.py --input_dir=./data/test/common_set --output_dir=./prep/test/common_set --istrain=False --img_size=112
python preprocessing.py --input_dir=./data/test/challenge_set --output_dir=./prep/test/challenge_set --istrain=False --img_size=112
python preprocessing.py --input_dir=./data/test/300w_private_set --output_dir=./prep/test/300w_private_set --istrain=False --img_size=112