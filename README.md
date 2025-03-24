# Aligning Logits Generatively for Principled Black-Box Knowledge Distillation

<div align="center">

<div>
    <a href='' target='_blank'>Jing Ma</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=-D5k5ioAAAAJ&view_op=list_works' target='_blank'>Xiang Xiang</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Ke Wang</a><sup>2</sup>&emsp;
    <a href='' target='_blank'>Yuchuan Wu</a><sup>2</sup>&emsp;
    <a href='' target='_blank'>Yongbin Li</a><sup>2</sup>
    
</div>
<div>
<sup>1</sup>School of Artificial Intelligence and Automation, Huazhong University of Science and Technology&emsp;

<sup>2</sup>Alibaba Group&emsp;
</div>
</div>

This repository contains the code for our CVPR 2024 paper:

> Aligning Logits Generatively for Principled Black-Box Knowledge Distillation

For more details, please refer to our [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Aligning_Logits_Generatively_for_Principled_Black-Box_Knowledge_Distillation_CVPR_2024_paper.html).


## Environment

This implementation is based on PyTorch, and we recommend using a GPU to run the experiments.

Install the required packages using `conda` and `pip`:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Run

### 1. Prepare a Teacher Model

If you don't already have a teacher model ready, you'll need to train one.

```shell
python -u project/teacher.py --model ResNet32

# or choose to run in the background monitor and output the logs
nohup python -u project/teacher.py --model ResNet32 > output.log 2>error.log &
```

### 2. Deprivatization

The first step is to train a DCGAN to accomplish the deprivatization, as mentioned in our our [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Aligning_Logits_Generatively_for_Principled_Black-Box_Knowledge_Distillation_CVPR_2024_paper.html).


```shell
python -u project/DCGAN.py --teacher ResNet32

# or choose to run in the background monitor and output the logs
nohup python -u project/DCGAN.py --teacher ResNet32 > output.log 2>error.log &
```

### 3. Distillation

The second step carries out black-box knowledge distillation, and you can experiment with different KD methods by changing the `{method}` parameter.

```shell
python -u project/{method}.py --teacher ResNet32 --model ResNet8

# or choose to run in the background monitor and output the logs
nohup python -u project/{method}.py --teacher ResNet32 --model ResNet8 > output.log 2>error.log &
```

`{method}` can be one of the following:
`KD`, `ML`, `AL`, `DKD`, `DAFL`, `KN`, `AM`, `DB3KD`, `MEKD`.

For `MEKD`, you can specify the `--res_type` parameter to choose the response type, which can be:
`soft`, `hard`.


## Citation

If you find our work helpful, please consider citing:

```
@inproceedings{ma2024aligning,
  title={Aligning Logits Generatively for Principled Black-Box Knowledge Distillation},
  author={Ma, Jing and Xiang, Xiang and Wang, Ke and Wu, Yuchuan and Li, Yongbin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23148--23157},
  year={2024}
}
```
