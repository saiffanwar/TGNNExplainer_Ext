# Download wikipedia and reddit datasets
Download from http://snap.stanford.edu/jodie/wikipedia.csv and http://snap.stanford.edu/jodie/reddit.csv and put them into ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/src/dataset/data


# Preprocess real-world datasets
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/src/models/ext/tgat
python process.py -d wikipedia
python process.py -d reddit

```

# Generate simulate dataset
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/src/dataset
python generate_simulate_dataset.py -d simulate_v1(simulate_v2)
```



# Generate explain indexs
```
cd  ~/workspace/GNNEXPLAINER-PUBLIC/tgnnexplainer/src/dataset
python tg_dataset.py -d wikipedia(reddit, simulate_v1, simulate_v2) -c index
```

# Train tgat/tgn model
tgat:
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/src/models/ext/tgat
./train.sh
./cpckpt.sh
```

tgn:
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/tgnnexplainer/src/models/ext/tgn
./train.sh
./cpckpt.sh
```

# Run our explainer and other  baselines
```
cd  ~/workspace/TGNNEXPLAINER-PUBLIC/benchmarks/explainer_run
./run.sh
``` 


