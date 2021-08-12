
# """============= Baseline Runs --- Online Products ===================="""
main=train_baseline
datapath=data
gpu=${1:-0}

python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_Npair --seed 0 --bs 112 --samples_per_class 2 --loss npair --batch_mining npair --arch resnet50_frozen

python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_GenLifted --seed 0 --bs 112 --samples_per_class 2 --loss lifted --batch_mining lifted --arch resnet50_frozen

python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_Histogram --seed 0 --bs 112 --samples_per_class 2 --loss histogram --arch resnet50_frozen_normalize --loss_histogram_nbins 11

python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_Contrastive --seed 0 --bs 112 --samples_per_class 2 --loss contrastive --batch_mining distance --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_Angular --seed 0 --bs 112 --samples_per_class 2 --loss angular --batch_mining npair --arch resnet50_frozen
 
python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_ArcFace --seed 0 --bs 112 --samples_per_class 2 --loss arcface --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_Triplet_Random --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining random --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_Triplet_Semihard --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_Triplet_Softhard --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining softhard --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_Triplet_Distance --seed 0 --bs 112 --samples_per_class 2 --loss triplet --batch_mining distance --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_Quadruplet_Distance --seed 0 --bs 112 --samples_per_class 2 --loss quadruplet --batch_mining distance --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 6 --source $datapath --n_epochs 100 --group SOP_Margin_b06_Distance --loss_margin_beta 0.6 --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 2 --source $datapath --n_epochs 100 --group SOP_Margin_b12_Distance --seed 0 --bs 112 --samples_per_class 2 --loss margin --batch_mining distance --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 2 --source $datapath --n_epochs 100 --group SOP_SNR_Distance  --seed 0 --bs 112 --samples_per_class 2 --loss snr --batch_mining distance --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 2 --source $datapath --n_epochs 100 --group SOP_MS --seed 0 --gpu 0 --bs 112 --samples_per_class 2 --loss multisimilarity --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 2 --source $datapath --n_epochs 100 --group SOP_Softmax --seed 0 --bs 112 --samples_per_class 2 --loss softmax --batch_mining distance --arch resnet50_frozen_normalize --loss_softmax_lr 0.002

python $main.py --dataset online_products --kernels 2 --source $datapath --n_epochs 100 --group SOP_ProxyNCA --seed 0 --bs 112 --samples_per_class 2 --loss proxynca --arch resnet50_frozen_normalize

python $main.py --dataset online_products --kernels 2 --source $datapath --n_epochs 100 --group SOP_SoftTriple --seed 0 --bs 32 --samples_per_class 2 --loss softtriplet --arch resnet50_frozen_normalize --loss_softtriplet_gamma 10 --loss_softtriplet_lambda 20 --loss_softtriplet_lrmulti 10 --lr 1e-1