### FSCIL3D-XL

The FSCIL3D-XL benchmark is programmatically constructed from publicly available datasets: the S2S module derives from **ShapeNet55** and **ModelNet40**, while the S2R module synthesizes **ShapeNet55** and **CO3D**. No original data collection or generation was conducted. This document will then provide detailed information on: 1.Download links for all open-source datasets. 2. Methods for application and extension.

### Download links for all open-resource datasets.

- **ModelNet40**  

  We use the download link in [ModelNet40_Align](https://github.com/lmb-freiburg/orion). You can download it according to the [official documentation](https://github.com/lmb-freiburg/orion/blob/master/datasets/get_modelnet40.sh).
  License: GPL-3.0 license

- **ShapeNet55**  

  We use the download link in [Point-Bert](https://github.com/Julie-tang00/Point-BERT). You can download the processed ShapeNet55/34 dataset at [[BaiduCloud](https://pan.baidu.com/s/16Q-GsEXEHkXRhmcSZTY86A)] (code:le04) or [[Google Drive](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing)].
  License: MIT License

- **Co3D (Common Objects in 3D)** 

  You can download the CO3D dataset through [Meta's official website of CO3D](https://ai.meta.com/datasets/co3d-downloads/ ). You can keep only the point cloud files, such as "2000.ply".
  License: Attribution-NonCommercial 4.0 International. You can find the detailed license in the [Meta's official website of CO3D](https://ai.meta.com/datasets/co3d-downloads/ )

### Methods for application and extension.
You can find and use our dataset construction code in the [our repository](https://github.com/HIT-leaderone/FILP-3D). Next, I will introduce how to use our preconfigured continual learning dataset and how to extend your own dataset.

#### How to use S2S and S2R

Fist, you should prepare the open-resource dataset and codes as follows:

```plain
│FILP-3D(or your code dir)/
├──datasets/
│   ├──CILdataset.py
│   ......
├──data/
│   ├──ModelNet40_manually_aligned/
│   ├──ShapeNet55/
│   ├──CO3D/
│   ......
├──session_settings.py
├──main.py
```

Then, you can use the session_maker in your main.py as follows:

```python
from session_settings import shapenet2modelnet, shapenet2co3d

session_maker = shapenet2co3d() ### you can change to shapenet2modelnet
id2name = session_maker.get_id2name()
io.cprint(session_maker.info())

dataset_train_0, dataset_test_0 = session_maker.make_session(session_id=0, update_memory=args.memory_shot)
num_cat_0 = dataset_test_0.get_cat_num()
train_loader_0 = torch.utils.data.DataLoader(dataset_train_0, batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=args.pin_memory, shuffle=True, persistent_workers=True)
test_loader_0 = torch.utils.data.DataLoader(dataset_test_0, batch_size=args.batch_size, num_workers=args.workers,
                                            pin_memory=args.pin_memory, shuffle=True, persistent_workers=True)
### Insert your base task code here ###

for task_id in range(1, session_maker.tot_session()):
    io.cprint("="*100, f"Task {task_id}", "="*100)
    runtime_stat['task_id'] = task_id
    dataset_train_i, dataset_test_i = session_maker.make_session(session_id=task_id, update_memory=args.memory_shot)
    num_cat_i = dataset_test_i.get_cat_num()
    train_loader_i = torch.utils.data.DataLoader(dataset_train_i, batch_size=args.batch_size, num_workers=args.workers,
                                                 pin_memory=args.pin_memory, shuffle=True, persistent_workers=True)
    test_loader_i = torch.utils.data.DataLoader(dataset_test_i, batch_size=args.batch_size, num_workers=args.workers,
                                                pin_memory=args.pin_memory, shuffle=True, persistent_workers=True)
    ### Insert your incremental task code here ###
```

#### How to extend your own dataset

First, I need to explain the functionality of each part of our code.

- CILdataset.py

  You need to write a class for the dataset you need. This class should handle reading and matching all point cloud file paths  with their labels, as well as provide a method to load point clouds based on their paths. Take `ModelNet40CIL` as an example.

```python
class ModelNet40AlignCIL(Dataset):
    cats = {'airplane': 0,...} ### Class and label mapping of the dataset
    id2name = list(cats.keys())
    def __init__(self, root='Your dataset path', partition='train', banlist=[]):
        assert partition in ('test', 'train')
        super().__init__()
        self.root = root
        self.partition = partition
        self._load_data(banlist)
	
    ### Read the point cloud paths and their corresponding categories sequentially in self.paths and self.labels lists
    def _load_data(self, banlist):
        self.paths = []
        self.labels = []
        for cat in os.listdir(self.root):
            if cat in banlist:
                continue
            cat_path = os.path.join(self.root, cat, self.partition)
            for case in os.listdir(cat_path):
                if case.endswith('.off'):
                    self.paths.append(os.path.join(cat_path, case))
                    self.labels.append(ModelNet40AlignCIL.cats[cat])
    
    ### You should write a load method for your point cloud data
    def get_load_method(self):
        def load(path, pt_num=1024):
            points = torch.Tensor(offread_uniformed(path, sampled_pt_num=pt_num)).type(torch.FloatTensor)
            rota1 = axis_angle_to_matrix(torch.tensor([0.5 * np.pi, 0, 0]))
            rota2 = axis_angle_to_matrix(torch.tensor([0, -0.5 * np.pi, 0]))
            points = points @ rota1 @ rota2
            return points.numpy()
        return load
    
    ### These two function are same but necessary
    def __getitem__(self, index):      
        return self.paths[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)
```

- session_settings.py

  The functions in this code determine which dataset serves as the base class and which dataset serves as the incremental class in your designed continual learning dataset. Take `shapenet2co3d` as a example:

```python
def shapenet2co3d():
    session_maker = SessionMaker()
    ### Create the 3D dataset classes you need. Some categories may be overlapping and need to be removed during the incremental phase. In such cases, you should add the categories you want to exclude to the ban_list.
    shapenet_train = ShapeNetCIL(partition='train')
    shapenet_test = ShapeNetCIL(partition='test')
    shapenet_id2name = ShapeNetCIL.id2name
    co3d = CO3DCIL(banlist=['car', 'microwave', 'bowl', 'bottle', 'skateboard', 'bench', 'motorcycle', 'laptop', 'chair'])
    co3d_id2name = co3d.get_label2name()
    ### Append the 3D dataset class to session_maker
    ### append_dataset_train_test: input train-test dataset
    ### append_dataset: only input one dataset, random split train-test data
    ### set_session: determine the number of base classes and the number of incremental classes in each session. The number of shot is 5, you can change it by "inc_few_shot". If the base-task is also few-shot, you can set "base_few_shot"(0 for represents full training).
    session_maker.append_dataset_train_test(shapenet_train, shapenet_test, shapenet_id2name) 
    session_maker.append_dataset(co3d, co3d_id2name, split_ratio=0.8) 
    session_maker.set_session(num_base_cat=55, num_inc_cat=4)
    return session_maker
```

Then, you can use your FSCIL dataset defined in the session_settings.py  as same as `shapenet2modelnet`