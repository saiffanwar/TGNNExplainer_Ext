# run all explainers
dataset=wikipedia # wikipedia, reddit, simulate_v1, simulate_v2
model=tgat # tgat, tgn

# ours
#python ~/PhD/TGNNExplainer/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=tgnnexplainer models=${model}

# SA explainer
python ~/PhD/TGNNExplainer/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=sa_explainer models=${model}

# baselines
#python ~/PhD/TGNNExplainer/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=attn_explainer_tg models=${model}
#python ~/PhD/TGNNExplainer/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=pg_explainer_tg models=${model}
#python ~/PhD/TGNNExplainer/benchmarks/explainer_run/run.py datasets=${dataset} device_id=0 explainers=pbone_explainer_tg models=${model}
