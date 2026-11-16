"""一次性自检脚本：打印 models/model.joblib 的实际结构（不修改任何文件）。"""
import warnings

import joblib
import numpy as np

warnings.simplefilter("ignore")

m = joblib.load("models/model.joblib")
print("pipeline steps:", [name for name, _ in m.steps])
for name, step in m.steps:
    print(f"\n--- {name}: {type(step).__module__}.{type(step).__name__}")
    if hasattr(step, "get_params"):
        p = step.get_params()
        for k in sorted(p):
            v = p[k]
            if isinstance(v, (int, float, str, bool, type(None), tuple)):
                print(f"    {k} = {v}")

last = m.steps[-1][1]
if hasattr(last, "classes_"):
    print("\nclasses_:", list(last.classes_))
if hasattr(last, "n_features_in_"):
    print("n_features_in_:", last.n_features_in_)
if hasattr(last, "n_support_"):
    print("n_support_ (每类支持向量数):", list(np.asarray(last.n_support_)))
    print("支持向量总数:", int(np.sum(last.n_support_)))
if hasattr(m, "n_features_in_"):
    print("pipeline n_features_in_:", m.n_features_in_)
