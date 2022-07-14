### Flexible Hybrid Lenses Light Field Super-Resolution using Layered Refinement

#### Requirements

- Ubuntu 20.04.3 LTS (GNU/Linux 5.11.0-40-generic x86_64)
- python==3.7.0
- pytorch==1.10.0+cu113
- numpy==1.21.4
- pandas==1.3.4
- opencv==3.4.2
- scikit-image==0.17.2
- scikit-learn==1.0.1

#### The computing infrastructure

- NVIDIA RTX A5000

## Testing with the pre-trained model

We provide the pre-trained model of $\times 4$ and $\times 8$ task.

The datasets should be prepared as the **micro-lens** image and the **PNG** format file. We provide a processed LF images for testing, which is `/Datasets/Antiques_dense.png`

```bash
python test.py --image_path ../Datasets/ --view_n_ori 9 --scale 8 --view_n 9 --disparity_range 4 --disparity_count 32 --is_save 1 --gpu_no 0
```

Use `python test.py -h` to get more helps.

