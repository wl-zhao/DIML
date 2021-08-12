dataset=${1:-cub200}
bs=${2:-112}
loss=${3:-margin_diml}
epochs=${4:-150}
seed=${5:-0}

python train_diml.py --dataset $dataset --loss $loss --batch_mining distance \
              --group ${dataset}_$loss --seed $seed \
              --bs $bs --data_sampler class_random --samples_per_class 2 \
              --arch resnet50_diml_frozen_normalize  --n_epochs $epochs \
              --lr 0.00001 --embed_dim 128 --evaluate_on_gpu
