data=../data/smiles_cas_N6512.smi
batch_size=128
dim=64
n_conv_hidden=4
n_mlp_hidden=1
dropout=0.1
lr=1e-4
n_epochs=1000
patience=1
model_path=../model/checkpoint_model.pth

python ./src/train.py $data $batch_size $dim $n_conv_hidden $n_mlp_hidden $dropout $lr $n_epochs $patience $model_path