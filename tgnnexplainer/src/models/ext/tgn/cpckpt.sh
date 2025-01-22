model=tgn
dataset=wikipedia # wikipedia, reddit, simulate_v1, simulate_v2
epoch=10


source_path=./saved_checkpoints/tgn-attn-${dataset}-${epoch}.pth
target_path=~/PhD/TGNNExplainer_Ext/tgnnexplainer/src/models/checkpoints/${model}_${dataset}_best.pth
cp ${source_path} ${target_path}
echo ${source_path} ${target_path} 'copied'
