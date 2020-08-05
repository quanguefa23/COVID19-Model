# Python SIR-Model Readme

## Install

Please follow tutorial at [link](pytorch.org) to install `PyTorch`.

Then install other package:
```
pip install numpy pandas
```

## Train model
All train source code are under `sir_model`.

To train SIR model:
```
python solver.py <path to csv data> <population of that countries>
```

To train SIRD model:
```
python solver_sird.py <path to csv data> <population of that countries>
```

## Get infer result:
```
python infer.py <path to csv data> <population of that countries> \
        --beta <beta value> \
        --gamma <gamma value> \
        --output_file output.csv 
```

## Get model performance result:
```
python calculate.py <path to output>
```
