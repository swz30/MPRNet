## Evaluation

- Download the [model](https://drive.google.com/file/d/1LODPt9kYmxwU98g96UrRA0_Eh5HYcsRw/view?usp=sharing) and place it in `./pretrained_models/`

#### Testing on SIDD dataset
- Download SIDD Validation Data and Ground Truth from [here](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php) and place them in `./Datasets/SIDD/test/`
- Run
```
python test_SIDD.py 
```
#### Testing on DND dataset
- Download DND Benchmark Data from [here](https://noise.visinf.tu-darmstadt.de/downloads/) and place it in `./Datasets/DND/`
- Run
```
python test_DND.py 
```

#### To reproduce PSNR/SSIM scores of the paper, run MATLAB script
```
evaluate_SIDD.m
```
