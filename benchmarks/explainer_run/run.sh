export PYTHONPATH=""
export PYTHONPATH="/home/saif/PhD/TGNNExplainer_Ext/tgnnexplainer"
#export PYTHONPATH="$PYTHONPATH:~/TGNNExplainer_Ext/tgnnexplainer/src"
echo "PYTHONPATH=$PYTHONPATH"
# run all explainers
dataset=${4} # wikipedia, reddit, simulate_v1, simulate_v2
model=${2} # tgat, tgn

if [ "$1" == "tgnne" ]; then
    python ~/PhD/TGNNExplainer_Ext/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=tgnnexplainer models=${model} results_batch=${3}
fi

# SA explainer
if [ "$1" == "sa" ]; then
    python3.12 ~/PhD/TGNNExplainer_Ext/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=sa_explainer models=${model} results_batch=${3}
fi

if [ "$1" == "temp_me" ]; then
    python3.12 ~/PhD/TGNNExplainer_Ext/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=temp_me models=${model} results_batch=${3}
fi
if [ "$1" == "pg" ]; then
    python3.12 ~/PhD/TGNNExplainer_Ext/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=pg_explainer_tg models=${model} results_batch=${3}
fi

#python ~/PhD/TGNNExplainer/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=pbone_explainer_tg models=${model}
# baselines
#python ~/PhD/TGNNExplainer/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=attn_explainer_tg models=${model}
