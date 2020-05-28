python DAN_V2.py -ds 2 --data_dir=./prep/test/common_set --data_dir_test=None -nlm 68 -mode predict
python move_result.py ./prep/predict common_set

python DAN_V2.py -ds 2 --data_dir=./prep/test/challenge_set --data_dir_test=None -nlm 68 -mode predict
python move_result.py ./prep/predict challenge_set

python DAN_V2.py -ds 2 --data_dir=./prep/test/300w_private_set --data_dir_test=None -nlm 68 -mode predict
python move_result.py ./prep/predict 300w_private_set