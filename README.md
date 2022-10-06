# Flexible Hybrid Lenses Light Field Super-Resolution using Layered Refinement

## Requirements

- Ubuntu 20.04.3 LTS (GNU/Linux 5.11.0-40-generic x86_64)
- python==3.7.0
- pytorch==1.10.0+cu113
- numpy==1.21.4
- pandas==1.3.4
- opencv==3.4.2
- scikit-image==0.17.2
- scikit-learn==1.0.1

## The computing infrastructure

- NVIDIA RTX A5000

## Testing with the pre-trained model

We provide the pre-trained model of $\times 4$ and $\times 8$ task.

Please first download our dataset via [BaiDu Drive](https://pan.baidu.com/s/1jZud3Jd3NodWc-zMrYNsBQ) (key: 2w7c), and place the folder `./LFHSR_Datasets/`.

Finally, you can test with the pre-trained model.
```bash
python test.py --datasets Datasets_S --view_n_ori 9 --scale 8 --view_n 9 --disparity_range 2 --disparity_count 32 --is_save 1 --gpu_no 0
```

If you want to test with your own datasets, you should put the LF images into the folder `./LFHSR_Datasets/Test/` and write the name of the LF images into the `./data_list/Datasets_yourselves.txt`.
Please note that the datasets should be prepared as the **micro-lens** image and the **PNG** format file.

Use `python test.py -h` to get more helps.

## Training

The training code is coming soon.