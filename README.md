
# Main Code Structured Repo

This repo is organized so you can run either model by name from `example.ipynb`.

Supported model names:
- `autodiff`
- `our_model`

Typical notebook inputs:
```python
MODELS_TO_RUN = ["autodiff", "our_model"]
CONFIGS = [{"NumAllGene": 2000, "NumTF": 1500, "B": 1}]
DATA_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
```

By default, the notebook shows the summary table and plots inline and does not save files.
