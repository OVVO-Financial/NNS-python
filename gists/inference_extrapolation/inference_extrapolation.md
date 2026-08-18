# Nonparametric vs. Tree-Based Models: Inference & Extrapolation Benchmark

This repository contains a rigorous comparative benchmark evaluating **NNS** (`nns_reg` and `nns_stack`) against traditional tree-based algorithms (**Random Forest** and **XGBoost**) across multiple mathematical functions and varying levels of Gaussian noise ($\sigma \in \{0.05, 0.10, 0.50\}$).


See the book for the theoretical support to this method: https://ovvo-financial.github.io/NNS/book/


## Objective

The core focus of this benchmark is to test two critical capabilities that standard tree architectures structurally fail to handle:
1. **In-Sample Inference (Interpolation):** Accuracy when predicting points located firmly inside the training support.
2. **Out-of-Sample Extrapolation:** Behavior when projecting trends beyond the final boundary of the training data.

---

## Experimental Setup

* **Training Range:** $x \in [0, 4\pi]$ ($N = 1000$ samples)
* **Inference Point:** $x = 2\pi$ (Inside support)
* **Extrapolation Point:** $x = 13.0$ (Outside support, extrapolation distance $\approx 0.4336$)
* **Functions Tested:**
  * **Quadratic:** $f(x) = 0.2x^2$
  * **Sine:** $f(x) = \sin(x)$
  * **Complex Growth:** $f(x) = x \sin(x)$

---

## Benchmark Results

| Function | Noise Std | Method | Inf Error | Ext Error | Time (sec) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Quadratic** | 0.05 | NNS (`nns_reg`) | 0.00452 | 0.36816 | 0.03285 |
| | 0.05 | NNS (`nns_stack`) | 0.00842 | 0.35068 | 1.60114 |
| | 0.05 | Random Forest | 0.01620 | 2.31585 | 0.17961 |
| | 0.05 | XGBoost | 0.06016 | 2.35611 | 0.06451 |
| | 0.10 | NNS (`nns_reg`) | 0.15035 | 1.64862 | 0.03413 |
| | 0.10 | NNS (`nns_stack`) | 0.08118 | 2.12266 | 1.62207 |
| | 0.10 | Random Forest | 0.13381 | 2.23735 | 0.18084 |
| | 0.10 | XGBoost | 0.14065 | 2.27693 | 0.06679 |
| | 0.50 | NNS (`nns_reg`) | 0.09111 | 6.49919 | 0.03595 |
| | 0.50 | NNS (`nns_stack`) | 0.06918 | 3.16209 | 1.54821 |
| | 0.50 | Random Forest | 0.29553 | 2.46085 | 0.17941 |
| | 0.50 | XGBoost | 0.16737 | 2.43229 | 0.06462 |
| **Sine** | 0.05 | NNS (`nns_reg`) | 0.00043 | 0.19648 | 0.01485 |
| | 0.05 | NNS (`nns_stack`) | 0.03717 | 0.08899 | 1.37911 |
| | 0.05 | Random Forest | 0.03694 | 0.42196 | 0.19416 |
| | 0.05 | XGBoost | 0.05868 | 0.42540 | 0.07167 |
| | 0.10 | NNS (`nns_reg`) | 0.00063 | 0.38897 | 0.02423 |
| | 0.10 | NNS (`nns_stack`) | 0.03072 | 0.67406 | 1.39635 |
| | 0.10 | Random Forest | 0.06317 | 0.47853 | 0.18452 |
| | 0.10 | XGBoost | 0.07191 | 0.46131 | 0.06517 |
| | 0.50 | NNS (`nns_reg`) | 0.07233 | 0.21179 | 0.02243 |
| | 0.50 | NNS (`nns_stack`) | 0.05696 | 0.50496 | 1.23556 |
| | 0.50 | Random Forest | 0.22463 | 0.53212 | 0.18647 |
| | 0.50 | XGBoost | 0.04897 | 0.53054 | 0.06416 |
| **x\*Sin(x)** | 0.05 | NNS (`nns_reg`) | 0.09719 | 0.00183 | 0.02579 |
| | 0.05 | NNS (`nns_stack`) | 0.01217 | 0.13343 | 1.61712 |
| | 0.05 | Random Forest | 0.02199 | 5.71817 | 0.18083 |
| | 0.05 | XGBoost | 0.14631 | 5.77525 | 0.06486 |
| | 0.10 | NNS (`nns_reg`) | 0.01124 | 0.62549 | 0.02385 |
| | 0.10 | NNS (`nns_stack`) | 0.01329 | 0.01588 | 1.62006 |
| | 0.10 | Random Forest | 0.02364 | 5.60796 | 0.18897 |
| | 0.10 | XGBoost | 0.19378 | 5.66961 | 0.06703 |
| | 0.50 | NNS (`nns_reg`) | 0.11526 | 1.71911 | 0.02658 |
| | 0.50 | NNS (`nns_stack`) | 0.29796 | 0.48497 | 1.32202 |
| | 0.50 | Random Forest | 0.56975 | 5.45844 | 0.18908 |
| | 0.50 | XGBoost | 0.03204 | 5.49367 | 0.06941 |

---

## Key Takeaways

1. **Structural Extrapolation Failure in Trees:** 
   Tree-based models (Random Forest and XGBoost) rely on terminal leaf averages. When tasked with predicting outside their training envelope (e.g., on $x \sin(x)$), they hit a hard mathematical wall and flatline, resulting in massive, static errors ($\approx 5.4$ to $5.7$).
   
2. **Native Geometric Projections:** 
   NNS utilizes localized partial moments rather than binary partitions. This allows `nns_reg` and `nns_stack` to capture underlying momentum and project continuous curves past the training boundary with minimal error (e.g., extrapolation error dropping as low as $0.0018$ on clean functions).

3. **Computational Efficiency:** 
   The base `nns_reg` implementation delivers lightning-fast performance, completing fits and predictions in **~0.02 to 0.03 seconds**—consistently outrunning both Random Forest and XGBoost. Meanwhile, `nns_stack` utilizes cross-validated dimensional ensembling to stabilize performance against heavy boundary noise.



## Code
```py
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Import NNS functions from the ovvo-nns package
from nns import nns_reg, nns_stack

# --- 1. Test Configurations ---
np.random.seed(42)

def f_sine(x): 
    return np.sin(x)

def f_quad(x): 
    return 0.2 * (x ** 2)

def f_complex(x): 
    return x * np.sin(x)

functions = {
    "Sine": f_sine,
    "Quadratic": f_quad,
    "x*Sin(x)": f_complex
}

noise_levels = [0.05, 0.1, 0.5]

# --- CORRECTED DOMAIN TO MATCH R SCRIPT ---
X_MIN, X_MAX = 0.0, 4 * np.pi
N_SAMPLES = 1000

x_inf_val = 2 * np.pi   # Inside support
x_ext_val = 13.0        # Outside support (extrapolation distance ≈ 0.4336)

x_train = np.linspace(X_MIN, X_MAX, N_SAMPLES)
x_inf = np.array([x_inf_val])
x_ext = np.array([x_ext_val])

# Standard reshaped arrays for tree-based models
X_tr_mat = x_train.reshape(-1, 1)
X_inf_mat = x_inf.reshape(-1, 1)
X_ext_mat = x_ext.reshape(-1, 1)

# The Copied Regressor Trick for NNS.stack
X2_train = np.column_stack((x_train, x_train))
X2_inf = np.column_stack((x_inf, x_inf))
X2_ext = np.column_stack((x_ext, x_ext))

# Bulletproof helper to extract the point estimate from NNS outputs
def get_nns_pred(res):
    if isinstance(res, dict):
        for key in ["Point.est", "Stack", "stack", "predictions"]:
            if key in res:
                return float(np.asarray(res[key])[0])
        raise KeyError(f"Missing prediction array in NNS output. Keys found: {list(res.keys())}")
    return float(np.asarray(res)[0])

results = []

# --- 2. Benchmark Execution ---
print(f"Training range: [{X_MIN}, {X_MAX:.4f}]")
print(f"Extrapolation point: {x_ext_val} (distance ≈ {x_ext_val - X_MAX:.4f})\n")
print("Running benchmarks across functions and noise levels...\n")

for func_name, func in functions.items():
    true_inf = func(x_inf_val)
    true_ext = func(x_ext_val)
    
    for noise in noise_levels:
        y_true = func(x_train)
        y_train = y_true + np.random.normal(0, noise, size=N_SAMPLES)
        
        # --- NNS (nns_reg) ---
        t0 = time.time()
        nns_reg_inf = get_nns_pred(nns_reg(x_train, y_train, point_est=x_inf))
        nns_reg_ext = get_nns_pred(nns_reg(x_train, y_train, point_est=x_ext))
        t_nns_reg = time.time() - t0
        
        results.append({
            "Function": func_name, "Noise_Std": noise, "Method": "NNS (nns_reg)",
            "Inf_Error": np.abs(nns_reg_inf - true_inf),
            "Ext_Error": np.abs(nns_reg_ext - true_ext),
            "Time_sec": t_nns_reg
        })
        
        # --- NNS (nns_stack) ---
        t0 = time.time()
        nns_stack_inf = get_nns_pred(nns_stack(X2_train, y_train, X2_inf, status=False))
        nns_stack_ext = get_nns_pred(nns_stack(X2_train, y_train, X2_ext, status=False))
        t_nns_stack = time.time() - t0
        
        results.append({
            "Function": func_name, "Noise_Std": noise, "Method": "NNS (nns_stack)",
            "Inf_Error": np.abs(nns_stack_inf - true_inf),
            "Ext_Error": np.abs(nns_stack_ext - true_ext),
            "Time_sec": t_nns_stack
        })
        
        # --- Random Forest ---
        t0 = time.time()
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42)
        rf.fit(X_tr_mat, y_train)
        rf_inf = rf.predict(X_inf_mat)[0]
        rf_ext = rf.predict(X_ext_mat)[0]
        t_rf = time.time() - t0
        
        results.append({
            "Function": func_name, "Noise_Std": noise, "Method": "Random Forest",
            "Inf_Error": np.abs(rf_inf - true_inf),
            "Ext_Error": np.abs(rf_ext - true_ext),
            "Time_sec": t_rf
        })
        
        # --- XGBoost ---
        t0 = time.time()
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror", max_depth=5, learning_rate=0.1, 
            n_estimators=100, random_state=42
        )
        xgb_model.fit(X_tr_mat, y_train)
        xgb_inf = xgb_model.predict(X_inf_mat)[0]
        xgb_ext = xgb_model.predict(X_ext_mat)[0]
        t_xgb = time.time() - t0
        
        results.append({
            "Function": func_name, "Noise_Std": noise, "Method": "XGBoost",
            "Inf_Error": np.abs(xgb_inf - true_inf),
            "Ext_Error": np.abs(xgb_ext - true_ext),
            "Time_sec": t_xgb
        })

# --- 3. Results Formatting ---
df = pd.DataFrame(results)
df_sorted = df.sort_values(by=["Function", "Noise_Std", "Method"])

pd.set_option('display.max_rows', None)
print(df_sorted.to_string(index=False, float_format=lambda x: f"{x:.5f}" if isinstance(x, float) else x))
```
