Have support for s3plr models

meta_audio_2 for all others
ssast2 for ssast
pase_plus for pase


python .\s3prl_main.py  --model_name tera --num_tasks 10000 --split test --classifier sklin --results_file wavlm_5s_10000_sklin_test --dims 1 --in_channels 1 --rep_length 5 --fine_tune False

added padding sorrt for s3plr
compartmentalised models