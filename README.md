# Implementation of VectorFloorSeg: Two-Stream Graph Attention Network for Vectorized Roughcast Floorplan Segmentation

No.of files in dir in terminal: ```ls -A | wc -l```

cubicasa5k
    |______colorful(276)
                |______10052(3)
                        |____F1_original.png
                        |____F1_scaled.png
                        |____model.svg
                |______10062(3)
                        |____F1_original.png
                        |____F1_scaled.png
                        |____model.svg
                
    |______high_quality(992)
        |______10010(3)
                    |____F1_original.png
                    |____F1_scaled.png
                    |____model.svg
        |______10014(3)
                |____F1_original.png
                |____F1_scaled.png
                |____model.svg

    |______high_quality_architectural(3732)
        |______10007(3)
                    |____F1_original.png
                    |____F1_scaled.png
                    |____model.svg
        |______10008(3)
                |____F1_original.png
                |____F1_scaled.png
                |____model.svg

    |______train.txt
    |______test.txt
    |______val.txt

Steps taken till now:

pip install cairosvg opencv-python triangle scikit-image matplotlib==3.0.3

1) Replace_with_CubiCasa/roughcast_data_generation.py => Takes in svg images and creates a roughcast vector image
2) Replace_with_CubiCasa/ImgRasterization.py          => Takes in roughcast vector image and creates a rasterization from it
3) Add "import lmdb" in Replace_with_CubiCasa/svg_loader.py 
4) Install Cubicasa repo, copy dataset from VecFloorSeg and create a conda env for it
5) Replace_with_CubiCasa/svg_loader.py => Processes floorplan images, creates segmentation labels, and saves the results as PNG files (wall_svg.png for wall segmentation and icon_svg.png for icon segmentation)

room_cls = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath", "Hallway", "Railing", "Storage", "Garage", "Other rooms"]
icon_cls = ["Empty", "Window", "Door", "Closet", "Electr. Appl.", "Toilet", "Sink", "Sauna bench", "Fire Place", "Bathtub", "Chimney"]

6) Copy the dataset from Cubicasa5k back to VecFloorSeg and move ImageProcessing_CubiCasa.py to VecFloorSeg home directory run:
python ImageProcessing_CubiCasa.py => Draw the wireframe of svg floorplan and turn the wireframe and image annotation into mmseg format

7)
```
torch_geometric/
├── graphgym/                 # GraphGym framework
│   ├── configs/             
│   │   └── CUBI.yaml        # Configuration for VecFloorSeg training
│   ├── models/              # Model definitions
│   │   ├── gnn.py          # GNN architecture
│   │   ├── encoder.py      # Node/edge feature encoders
│   │   └── head.py         # Output heads
│   ├── train.py            # Main training script
│   ├── eval.py             # Evaluation script
│   └── checkpoint.py        # Model checkpointing
│
├── nn/                      # Neural network modules
│   ├── conv/               
│   │   └── gat_conv.py     # Graph Attention Convolution
│   └── models/
│       └── attention.py    # Attention mechanisms
│
└── data/                    # Data handling
    ├── dataset.py          # Dataset classes
    └── dataloader.py       # Data loading utilities
```
torch_geometric/graphgym/train.py => To train model 

graphgym/models/gnn.py:

```
VecFloorSeg
    |____torch_geometric
        |____data
        |____graphgym
            |____models
                |____ __init__.py
                |____ act.py
                |____ encoder.py
                |____ gnn.py
                |____ head.py
                |____ layer.py
                |____ pooling.py
                |____ transform.py
                |____ .py
                |____ .py
            |____contrib
            |____utils
        |____io
        |____nn
        |____utils
        |____visualization
```








### Environment
    conda install --yes --file requirements.txt
- Install pyg following the instruction from [official site](https://pytorch-geometric.readthedocs.io/en/latest/), 
we recommend pyg==2.0.4
### Data preparation
- Download our processed data: [here](https://drive.google.com/drive/folders/1Rye_6crjcuII2LVaIwh4iDNowFqLp1Q6?usp=sharing)

### Pretrained backbone downloaded
    mkdir models
    cd models
Download ResNet-101 from pytorch official site [here](https://download.pytorch.org/models/resnet101-63fe2227.pth), rename to resnet101-torch.pth and move to models.

### Code preparation

- Replace the *graphgym* and *torch_geometric* in pyg with corresponding dir in our repository

### Train 

    python graphgym/train.py --cfg graphgym/configs/CUBI.yaml --seed 0
### Eval

    python graphgym/eval.py --cfg graphgym/configs/CUBI.yaml --eval train.epoch_resume 1 \
                              train.ckpt_prefix best val.extra_infos True seed 0

### Optional: Processing svg format dataset from CubiCasa-5k source data
**Notice: before running the code, please change the data dir within the code into your souce data dir**
- Source data downloaded: [here](https://zenodo.org/records/2613548)
- Download CubiCasa-5k source code and configure the environment: [here](https://github.com/CubiCasa/CubiCasa5k/tree/master)
- Put **_Replace_with_Cubicasa_** into CubiCasa-5k code repo
- Process source model.svg into roughcast svg format floorplans:


    python Replace_with_CubiCasa/roughcast_data_generation.py
  
- Render svg floorplan into image:


    python Replace_with_CubiCasa/ImgRasterization.py
  
- Render image annotation of floorplans:


    python Replace_with_CubiCasa/svg_loader.py
  
- Draw the wireframe of svg floorplan and turn the wireframe and image annotation into mmseg format:


    python DataPreparation/ImageProcessing_CubiCasa.py
  
- Process svg floorplan as pickle file:
    

    python SvgProcessing_CubiCasa.py
  

