import pandas as pd, numpy as np
import gzip, json, yaml, random, torch

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def load_yaml(fpath):
    with open(fpath, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    """
    Fix seed for reproducibility.
    Applies to Python, NumPy, PyTorch (CPU/GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
