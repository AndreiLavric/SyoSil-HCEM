# SyoSil-HCEM

Hardware Cost Estimator Model (HCEM) is a Python library which can be used to obtained the hardware cost of running a neural network algorithm on a specific hardware platform.


## Running an example

From the top-level directory execute the following commnd:

```bash
'python3 main.py -m super_resolution.onnx -c ibex'
```

All NN models should be placed inside the nn_models directory.

The default hardware platform is the Ibex core.