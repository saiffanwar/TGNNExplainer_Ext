export PYTHONPATH=""
export PYTHONPATH="/home/saif/PhD/TGNNExplainer_Ext/tgnnexplainer"
export PYTHONPATH="$PYTHONPATH:/home/saif/PhD/TGNNExplainer_Ext/tgnnexplainer/src"
echo "PYTHONPATH=$PYTHONPATH"
# run all explainers
dataset=wikipedia # wikipedia, reddit, simulate_v1, simulate_v2
model=tgat # tgat, tgn

if [ "$1" == "tgnne" ]; then
    python ~/PhD/TGNNExplainer_Ext/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=tgnnexplainer models=${model}
fi

# SA explainer
if [ "$1" == "sa" ]; then
    python ~/PhD/TGNNExplainer_Ext/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=sa_explainer models=${model}
fi

# baselines
#python ~/PhD/TGNNExplainer/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=attn_explainer_tg models=${model}
#python ~/PhD/TGNNExplainer/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=pg_explainer_tg models=${model}
#python ~/PhD/TGNNExplainer/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=pbone_explainer_tg models=${model}
