model=tgat
dataset=reddit
epoch=5

source_path=./saved_checkpoints/${dataset}-attn-prod-${epoch}.pth
target_path=~/PhD/TGNNExplainer_Ext/tgnnexplainer/src/models/checkpoints/${model}_${dataset}_best.pth
cp ${source_path} ${target_path}

echo ${source_path} ${target_path} 'copied'
