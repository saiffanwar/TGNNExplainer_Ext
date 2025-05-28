# dataset: wikipedia, reddit, simulate_v1, simulate_v2
export PYTHONPATH=""
export PYTHONPATH="/home/saif/PhD/TGNNExplainer_Ext/tgnnexplainer"
export PYTHONPATH="$PYTHONPATH:/home/saif/PhD/TGNNExplainer_Ext/tgnnexplainer/src"
echo "PYTHONPATH=$PYTHONPATH"


for i in 0
do
    echo "${i}-th run\n"

#    dataset=simulate_v1
#    python train_simulate.py -d ${dataset} --prefix tgn-attn --n_runs 1 --n_epoch 100 --n_layer 2 --n_degree 10 --use_memory --memory_update_at_end --gpu 0 \
#    --memory_dim 4 # memory_dim should equal to node/edge feature dim

    dataset=independent_cascade
    python train_self_supervised.py -d ${dataset} --prefix tgn-attn --n_runs 1 --n_epoch 100 --n_layer 2 --n_degree 10 --gpu 0

done













