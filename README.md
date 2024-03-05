## Requirements

### Installation

PyTorch, PyTorch3d, CLIP, pointnet2_ops, etc., are required. We recommend to create a conda environment and install dependencies in Linux as follows:

```
# create a conda environment
conda create -n clip2point python=3.7 -y
conda activate clip2point

# install pytorch & pytorch3d
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# install CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# install pointnet2 & other packages
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install -r requirements.txt
```

### Data preparation

The overall directory structure should be:

```
│CLIP2Point/
├──datasets/
├──data/
│   ├──ModelNet40_Align/
│   ├──ModelNet40_Ply/
│   ├──Rendering/
│   ├──ShapeNet55/
│   ......
├──.......
```

Moreover please download **CO3D** dataset in "/data/CO3D"

Please refer to [CLIP2Point](https://github.com/tyhuang0428/CLIP2Point) for the dataset download.

## Get start

download the pre-trained checkpoint [best_eval.pth](https://drive.google.com/file/d/1ZAnIANNMqRRRmaVtk8Kp93s_NkGU51zv/view?usp=sharing)  [best_test.pth](https://drive.google.com/file/d/1Jr1yXOu1yKmMs8K7XD8FnttPRHnZOZHx/view?usp=sharing) and  [dgcnn_occo_cls.pth](https://drive.google.com/file/d/1EG7zh8J_IE4rN9aNb_z7ePkAIwD9SwfB/view?usp=drive_link)

```
│FILP-3D/
├──pre_builts/
│   ├──vit32/
│   |	├──best_eval.pth/
│   |	├──best_test.pth/
│   ├──point/
│   |	├──dgcnn_occo_cls.pth/
```

```
python main.py
```

You can change session_settings.py and args to run in other datasets.