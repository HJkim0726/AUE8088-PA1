import subprocess
import yaml
import csv
import os

# 모델 리스트 정의
# MODEL_LIST = ["resnet18", "alexnet", "MyNetwork", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", 
#               "efficientnet_b6", "efficientnet_b7", "efficientnet_v2_l", "efficientnet_v2_m", "efficientnet_v2_s", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]
MODEL_LIST = ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]

for model_name in MODEL_LIST:
    print(f"▶ Training model: {model_name}")

    # config.py 파일의 MODEL_NAME만 교체
    with open("src/config.py", "r") as file:
        config_lines = file.readlines()

    with open("src/config.py", "w") as file:
        for line in config_lines:
            if line.strip().startswith("MODEL_NAME"):
                file.write(f"MODEL_NAME          = '{model_name}'\n")
            else:
                file.write(line)

    # train.py 실행
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Error training {model_name}:")
        print(result.stderr)
        continue

# OPTIMIZER_PARAMS_LIST    = [{'type': 'SGD', 'lr': 0.001, 'momentum': 0.9}, {'type': 'Adam', 'lr': 0.001}, {'type': 'AdamW', 'lr': 0.001}, {'type': 'RMSprop', 'lr': 0.001}]

# for optim_name in OPTIMIZER_PARAMS_LIST:
#     print(f"▶ Training model: {optim_name}")

#     with open("src/config.py", "r") as file:
#         config_lines = file.readlines()

#     with open("src/config.py", "w") as file:
#         for line in config_lines:
#             if line.strip().startswith("OPTIMIZER_PARAMS"):
#                 file.write(f"OPTIMIZER_PARAMS          = {optim_name}\n")
#             else:
#                 file.write(line)

#     # train.py 실행
#     result = subprocess.run(["python", "train.py"], capture_output=True, text=True)

#     if result.returncode != 0:
#         print(f"❌ Error training {optim_name}:")
#         print(result.stderr)
#         continue

# SCHEDULER_PARAMS_LIST = [{'type': 'MultiStepLR', 'milestones': [15, 30], 'gamma': 0.2}, {'type': 'MultiStepLR', 'milestones': [30, 60, 90], 'gamma': 0.5}, {'type': 'MultiStepLR', 'milestones': [10, 20], 'gamma': 0.1}]


# for scheduler_name in SCHEDULER_PARAMS_LIST:
#     print(f"▶ Training model: {scheduler_name}")

#     with open("src/config.py", "r") as file:
#         config_lines = file.readlines()

#     with open("src/config.py", "w") as file:
#         for line in config_lines:
#             if line.strip().startswith("SCHEDULER_PARAMS"):
#                 file.write(f"SCHEDULER_PARAMS          = {scheduler_name}\n")
#             else:
#                 file.write(line)

#     # train.py 실행
#     result = subprocess.run(["python", "train.py"], capture_output=True, text=True)

#     if result.returncode != 0:
#         print(f"❌ Error training {scheduler_name}:")
#         print(result.stderr)
#         continue





