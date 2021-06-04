# NIPS 2021 #2271

This temporary repository is the implementation of NIPS 2021 #2271 submission. An official repository will be released after review process.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Pretrained full group sparse model

To reproduce the results shown in the main body of paper, we provide our pretrained full group sparse models by HSPG. 

They can be downloaded via this [link](drive.google.com/drive/folders/1Zjj6-o6PM3F4VkFLpvg6-bBkpaQ4uiD9?usp=sharing).

These full group sparse models can be directly pruned by the following scripts to generate slimmer models with the exact same performance as the full group sparse models. 


## Pruning

To prune the full group sparse CNN models and construct the slimmer pruned models, run the this command:
+ backend: [vgg16|vgg16\_bn|resnet50]
+ dataset\_name: [cifar10|imagenet]
```prune
python prune/prune_cnn.py --backend <backend> --dataset_name <dataset_name>--checkpoint <checkpoint>
```
For examples,
- VGG16 on CIFAR10
```
python prune/prune_cnn.py --backend vgg16 --dataset_name cifar10 --checkpoint checkpoints/vgg16_cifar10_group_sparse.pt
```
- VGG16-BN on CIFAR10
```
python prune/prune_cnn.py --backend vgg16_bn --dataset_name cifar10 --checkpoint checkpoints/vgg16_bn_cifar10_group_sparse.pt
```
- ResNet50 on CIFAR10
```
python prune/prune_cnn.py --backend resnet50 --dataset_name cifar10 --checkpoint checkpoints/resnet50_cifar10_group_sparse.pt
```
- ResNet50 on ImageNet
```
python prune/prune_cnn.py --backend resnet50 --dataset_name imagenet --checkpoint checkpoints/resnet50_imagenet_group_sparse.pt
```



To prune the full group sparse Bert models and construct the slimmer pruned Berts, run the this command:
```
python prune/prune_bert_squad.py --checkpoint_dir <checkpoint_dir> --eval_data <data_file_path>
```
For example,
```
python prune/prune_bert_squad.py --checkpoint_dir checkpoints/bert_squad_oto_params_40_exact_71_f1_81 --eval_data data/squad/dev-v1.1.json

python prune/prune_bert_squad.py --checkpoint_dir checkpoints/bert_squad_oto_params_53_exact_71_f1_82 --eval_data data/squad/dev-v1.1.json

python prune/prune_bert_squad.py --checkpoint_dir checkpoints/bert_squad_oto_params_67_exact_72_f1_82 --eval_data data/squad/dev-v1.1.json

python prune/prune_bert_squad.py --checkpoint_dir checkpoints/bert_squad_oto_params_76_exact_72_f1_82 --eval_data data/squad/dev-v1.1.json

python prune/prune_bert_squad.py --checkpoint_dir checkpoints/bert_squad_oto_params_91_exact_75_f1_84 --eval_data data/squad/dev-v1.1.json
```

The above pruning script generates corresponding pruned models in checkpoints dir, which return the exact same output as the full group sparse models.


## Training
To train the CNNs in the paper, run this command:
```train
python train/run_cnn.py --opt train/configs/config_vgg16_hspg.yml

python train/run_cnn.py --opt train/configs/config_vgg16_bn_hspg.yml

python train/run_cnn.py --opt train/configs/config_resnet50_cifar10_hspg.yml

python train/run_cnn.py --opt train/configs/config_resnet50_imagenet_hspg.yml
```

To train the Berts in the paper, run this command:
```train
python train/run_bert_squad.py --opt train/configs/config_bert_squad_hspg_upgs_0.1.yml

python train/run_bert_squad.py --opt train/configs/config_bert_squad_hspg_upgs_0.3.yml

python train/run_bert_squad.py --opt train/configs/config_bert_squad_hspg_upgs_0.5.yml

python train/run_bert_squad.py --opt train/configs/config_bert_squad_hspg_upgs_0.7.yml

python train/run_bert_squad.py --opt train/configs/config_bert_squad_hspg_upgs_0.9.yml
```

## Results

Our pruned models achieve the following performance on :

### [Pruned VGG16 on CIFAR10]

|   FLOPs  | # of Params |  Top 1 Accuracy  |
| -------- | ----------- | ---------------- |
|   16.3%  |     2.5%    |      91.0%       |

### [Pruned VGG16-BN on CIFAR10]

|   FLOPs  | # of Params |  Top 1 Accuracy  |
| -------- | ----------- | ---------------- |
|   26.8%  |     5.5%    |      93.3%       |

### [Pruned ResNet50 on CIFAR10]

|   FLOPs  | # of Params |  Top 1 Accuracy  |
| -------- | ----------- | ---------------- |
|   12.8%  |     8.8%    |      94.4%       |


### [Pruned ResNet50 on ImageNet]

|   FLOPs  | # of Params |  Top 1 Accuracy  | Top 5 Accuracy |
| -------- | ----------- | ---------------- | -------------- |
|   34.5%  |    35.5%    |      74.7%       |      92.1%     |


### [Pruned Berts on SQuAD]

| # of Params |    Exact    |   F1-Score  |
| ----------- | ----------- | ----------- |
|    91.0%    |    75.0%    |    84.1%    |
|    76.2%    |    72.3%    |    82.1%    |
|    66.7%    |    71.9%    |    82.0%    |
|    53.3%    |    71.4%    |    81.5%    |
|    40.0%    |    70.9%    |    81.1%    |

## Contributing

The source code for the site is licensed under the MIT license.
