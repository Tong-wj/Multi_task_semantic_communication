DATASET=nyud/resnet50

MODEL=multi_task_baseline

python main.py --config_env configs/env.yml --config_exp configs/$DATASET/$MODEL.yml
