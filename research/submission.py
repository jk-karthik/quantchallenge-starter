import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import xgboost as xgb
import warnings


warnings.filterwarnings("ignore")



TRAIN_PATH = '/Users/jandh/Desktop/Old Desktop/od/Summer/quantchallenge-starter/train.csv'
TEST_PATH  = '/Users/jandh/Desktop/Old Desktop/od/Summer/quantchallenge-starter/test.csv'


VOL_COLS = ["A", "I", "F", "B", "D", "K", "L"]
LAGS       = [1, 5, 10, 20, 50]
ROLL_WIN   = [5, 10, 20, 50]
MAKE_DELTAS = True     # x - x_lag1 and x - x_lag5
MAKE_PCT    = True 
vol_input_columns = ['A','B','D','F','I','K','L']
return_input_columns = ['C','E','G','H','J','M','N']
input_columns = [chr(i) for i in range(65,79)]
TARGETS    = ["Y1", "Y2"]


VAL_SIZE   = 10_000    # last-block size
PURGE_GAP  = 500       # drop rows just before validation

SEEDS_LGB  = [42, 202, 909, 131, 777]     # seed averaging
SEEDS_XGB  = [42, 202, 909, 131, 777]   


# SEEDS_LGB  = [42]     # seed averaging
# SEEDS_XGB  = [42]  

LGB_PARAMS = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=16,
        min_data_in_leaf=1500,
        feature_fraction=0.60,
        bagging_fraction=0.60,
        bagging_freq=1,
        lambda_l2=40.0,
        extra_trees=True,
        verbosity=-1
    )

XGB_PARAMS = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.01,
        max_depth=5,
        colsample_bytree=0.50,
        subsample=0.50,
        tree_method="hist",
        verbosity=0,
        verbose_eval = False
    )



# ---------------- Train base XGB models on holdout ----------------
def train_xgb_holdout(X, y, tr_idx, va_idx, seeds, params):
    evals_result = {}
    preds_val = np.zeros(len(va_idx))
    dtrain = xgb.DMatrix(X[tr_idx], y[tr_idx])
    dval = xgb.DMatrix(X[va_idx], y[va_idx])
    best_iters = []
    for sd in seeds:
        m = xgb.train(
            dict(params, seed=sd),
            dtrain,
            num_boost_round=5_000,
            evals=[[dtrain,'train'],[dval,'valid']],
            evals_result = evals_result,
            callbacks=[xgb.callback.EarlyStopping(500)],
            verbose_eval=False
        )
        preds_val += m.predict(xgb.DMatrix(X[va_idx]), iteration_range=(0, m.best_iteration + 1)) / len(seeds)
        best_iters.append(m.best_iteration)
        print(
        f" XGB [Seed {sd}] Iter {m.best_iteration:5d} | "
        f"Train RMSE: {evals_result['train']['rmse'][m.best_iteration-1]:.6f} | "
        f"Valid RMSE: {evals_result['valid']['rmse'][m.best_iteration-1]:.6f}"
    )
    return preds_val, int(np.mean(best_iters))


# ---------------- Train base LGB models on holdout ----------------
def train_lgb_holdout(X, y, tr_idx, va_idx, seeds, params):
    preds_val = np.zeros(len(va_idx))
    dtrain = lgb.Dataset(X[tr_idx], y[tr_idx])
    deval  = lgb.Dataset(X[va_idx], y[va_idx])
    best_iters = []
    for sd in seeds:
        m = lgb.train(
            dict(params, seed=sd),
            dtrain,
            num_boost_round=5_000,
            valid_sets=[dtrain, deval],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(500),lgb.log_evaluation(0)]
        )
        preds_val += m.predict(X[va_idx], num_iteration=m.best_iteration) / len(seeds)
        best_iters.append(m.best_iteration)
        train_rmse = m.best_score["train"]["rmse"]
        valid_rmse = m.best_score["valid"]["rmse"]
        print(f"LGB [Seed {sd}] Iter {m.best_iteration:5d} | "
              f"Train RMSE: {train_rmse:.6f} | Valid RMSE: {valid_rmse:.6f}")
    return preds_val, int(np.mean(best_iters))


def best_weight(y_true, p1, p2):

    best_a, best_r2 = 1.0, r2_score(y_true, p1)
    for a in np.linspace(0.0, 1.0, 21):  # step=0.05
        mix = a * p1 + (1 - a) * p2
        r2 = r2_score(y_true, mix)
        if r2 > best_r2:
            best_r2, best_a = r2, a
    return best_a, best_r2


def create_input_features(columns,df,lag_dict = None):
    feature_list = []
    # print(lag_dict,'R' in lag_dict)
    for col in columns:
        feature_list.append(df[col])
        if col in return_input_columns and 'R' in lag_dict:
            for lag in range(lag_dict['R'][0],lag_dict['R'][1],1):
                if lag == 0 :
                    continue
                feature_list.append(df[col].shift(-lag).rename(f'{col}_{lag}'))

        if col in return_input_columns and 'R2' in lag_dict:
            for lag in range(lag_dict['R2'][0],lag_dict['R2'][1],1):
                # if lag == 0 :
                #     continue
                feature_list.append((df[col]**2).shift(-lag).rename(f'{col}_squared_{lag}'))
        
        
        if col in vol_input_columns and 'V' in lag_dict:
            for lag in range(lag_dict['V'][0],lag_dict['V'][1],1):
                if lag == 0 :
                    continue
                feature_list.append(df[col].shift(-lag).rename(f'{col}_{lag}'))
        
    feature_df = pd.concat(feature_list,axis=1)
    return feature_df



def engineer_time_features(
    df , 
    columns,
    lags,
    windows,
    add_deltas = True,
    add_pct = True,
    eps: float = 1e-6,
) :
    """
    Create lag, rolling, delta, and percent-change style features for selected columns.

    Returns
    -------
    DataFrame
        Original df joined with new feature columns.
    """
    new_cols = {}
    for c in columns:
        # Lags
        for lag in lags:
            new_cols[f"{c}_lag{lag}"] = df[c].shift(lag)
        # Rolling mean/std
        for w in windows:
            new_cols[f"{c}_roll{w}_mean"] = df[c].rolling(w).mean()
            new_cols[f"{c}_roll{w}_std"]  = df[c].rolling(w).std()
        # Short term deltas
        if add_deltas:
            new_cols[f"{c}_d1"] = df[c] - df[c].shift(1)
            new_cols[f"{c}_d5"] = df[c] - df[c].shift(5)

        # Percent change (to normalize level drift)
        if add_pct:
            lag1 = df[c].shift(1)
            new_cols[f"{c}_pct1"] = (df[c] - lag1) / (lag1.abs() + eps)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def predict_lgb(X, y, X_test, seeds, params, num_rounds):
    preds = np.zeros(len(X_test))
    for sd in seeds:
        m = lgb.train(dict(params, seed=sd), lgb.Dataset(X, y), num_boost_round=num_rounds)
        preds += m.predict(X_test) / len(seeds)
    return preds

def predict_XGB(X, y, X_test, seeds, params, approx_iters):
    preds = np.zeros(len(X_test))
    for sd in seeds:
        m = XGBRegressor(**dict(params, random_seed=sd))
        m.set_params(iterations=int(approx_iters))
        m.fit(X, y, verbose=False)
        preds += m.predict(X_test) / len(seeds)
    return preds

if __name__=="__main__":

    # ---------------- Load data ----------------
    train_data = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    
    # ---------------- Feature engineering for R^2 ----------------
    X_new = create_input_features(input_columns,train_data,{'R2': (-5,1)})

    ## Add time-series features like lags, rollings, deltas, pct

    X_new = engineer_time_features(X_new, input_columns, LAGS, ROLL_WIN, MAKE_DELTAS, MAKE_PCT)

    train_data_clean = pd.concat([X_new,train_data[TARGETS]],axis=1)


    n = train_data_clean.shape[0]
    val_start = n - VAL_SIZE
    train_end = max(0, val_start - PURGE_GAP)
    tr_idx = np.arange(0, train_end)
    va_idx = np.arange(val_start, n)
    print(f"[Split] Train={len(tr_idx)} | Purge={PURGE_GAP} | Valid={len(va_idx)}")

    updated_input_columns = list(train_data_clean.drop(columns=['Y1','Y2']).columns)

    X_all = train_data_clean[updated_input_columns].values
    Y_all = train_data_clean[TARGETS].values
    

        


    hold_lgb = {}
    best_iters_lgb = {}
    hold_xgb = {}
    best_iters_xgb = {}
    for t_idx, tgt in enumerate(TARGETS):
        y = Y_all[:, t_idx]
        p_lgb, bi_lgb = train_lgb_holdout(X_all, y, tr_idx, va_idx, SEEDS_LGB, LGB_PARAMS)
        hold_lgb[tgt] = p_lgb
        best_iters_lgb[tgt] = bi_lgb


        p_xgb, bi_xgb = train_xgb_holdout(X_all, y, tr_idx, va_idx, SEEDS_LGB, XGB_PARAMS)
        hold_xgb[tgt] = p_xgb
        best_iters_xgb[tgt] = bi_xgb

        # Print single-model holdout scores
        r2_lgb = r2_score(Y_all[va_idx, t_idx], hold_lgb[tgt])
        msg = f"[Holdout] {tgt} LGB R²={r2_lgb:.4f} , (best_iter≈{bi_lgb})"

        # XHB
        r2_xgb = r2_score(Y_all[va_idx, t_idx], hold_xgb[tgt])
        msg += f" | XGB R²={r2_xgb:.4f} , (iters≈{bi_xgb})"
        print(msg)


    ## Ensemble
    blend_alpha = {}  # yhat = alpha*LGB + (1-alpha)*XGB
    for t_idx, tgt in enumerate(TARGETS):
        yv = Y_all[va_idx, t_idx]
        p1 = hold_lgb[tgt]
        p2 = hold_xgb[tgt]
        a, r2b = best_weight(yv, p1, p2)
        blend_alpha[tgt] = a

        print(f"[Ensemble] {tgt}: alpha(LGB)={a:.2f} , R²={r2b:.4f}")

    mean_holdout = np.mean([
        r2_score(Y_all[va_idx, i],
                (blend_alpha[TARGETS[i]] * hold_lgb[TARGETS[i]] +
                (1 - blend_alpha[TARGETS[i]]) * (hold_xgb[TARGETS[i]] )))
        for i in range(len(TARGETS))
    ])
    print(f"\n=== Ensemble Mean R² : {mean_holdout:.4f} ===\n")

    X_test = create_input_features(input_columns,test,{'R2': (-5,1)})

    ## Add time-series features like lags, rollings, deltas, pct

    X_test = engineer_time_features(X_test, input_columns, LAGS, ROLL_WIN, MAKE_DELTAS, MAKE_PCT)

    assert X_test.shape[1] == X_all.shape[1], f"Test features {X_test.shape[1]} != Train features {X_all.shape[1]}"

    preds_test = np.zeros((len(X_test), len(TARGETS)))
    for t_idx, tgt in enumerate(TARGETS):
        y = Y_all[:, t_idx]
        num_rounds_lgb = best_iters_lgb[tgt]  
        p_lgb = predict_lgb(X_all, y, X_test, SEEDS_LGB, LGB_PARAMS, num_rounds_lgb)
        print("Predict LGB done for",tgt)

        approx_iters = best_iters_xgb[tgt] if best_iters_xgb.get(tgt) else 3000
        p_xgbt = predict_XGB(X_all, y, X_test, SEEDS_XGB, XGB_PARAMS, approx_iters)
        print("Predict XGBT done for",tgt)
        a = blend_alpha[tgt]
        preds_test[:, t_idx] = a * p_lgb + (1 - a) * p_xgbt
        print("Ensemble done for",tgt)

    # ---------------- Submission ----------------
    sub = pd.DataFrame({
        "id": np.arange(1, len(X_test) + 1, dtype=int),
        "Y1": preds_test[:, 0].astype(float),
        "Y2": preds_test[:, 1].astype(float),
    })
    assert sub.columns.tolist() == ["id", "Y1", "Y2"]
    assert len(sub) == len(test)
    assert np.isfinite(sub[["Y1","Y2"]].values).all()

    sub.to_csv("submission_final.csv", index=False)
    print("Saved submission_final.csv")



    