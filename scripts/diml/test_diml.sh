dataset=${1:-cub200}
embed_dim=${2:-128}
arch=${3:-resnet50_frozen_normalize}

python test_diml.py --dataset $dataset \
              --seed 0 --bs 16 --data_sampler class_random --samples_per_class 2\
              --arch $arch \
              --embed_dim $embed_dim --evaluate_on_gpu \
