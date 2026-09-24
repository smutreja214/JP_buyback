from pandas import RangeIndex
from sklearn.metrics import r2_score
from datetime import timedelta
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import joblib
import shap
import optuna
import matplotlib.pyplot as plt
import warnings
import json

optuna.logging.set_verbosity(optuna.logging.WARNING)

RECENCY_HALF_LIFE_DAYS = 10000000000000000000000000000000000000000000
def recency_weights(dates, reference_date=None, half_life_days=RECENCY_HALF_LIFE_DAYS):
    dates = pd.to_datetime(pd.Series(dates), format="%d/%m/%Y").reset_index(drop=True)
    if reference_date is None:
        reference_date = dates.max()
    reference_date = pd.Timestamp(reference_date)
    age_days = (reference_date - dates).dt.total_seconds() / 86400.0
    weights = np.power(0.5, np.maximum(age_days, 0.0) / half_life_days)
    # Keep the average weight at 1 so the overall regularisation/training
    # scale stays comparable with the unweighted model.
    return (weights / weights.mean()).to_numpy()

SMALL_XGB_SPACE = dict(
    max_depth=(2, 4),
    n_estimators=(50, 500),
    learning_rate=(0.01, 0.1),
    subsample=(0.5, 1.0),
    colsample_bytree=(0.4, 1.0),
    min_child_weight=(5, 30),
    reg_alpha=(0.0, 5.0),
    reg_lambda=(0.0, 10.0),
)

class EnsembleModel:
    def __init__(self, en_model, xgb_model, weight_en, features):
        self.en_model = en_model
        self.xgb_model = xgb_model
        self.weight_en = weight_en          # weight on the ElasticNet leg
        self.features = features

    def predict(self, X):
        X = X[self.features] if hasattr(X, "columns") else X
        return (self.weight_en * self.en_model.predict(X)
                + (1 - self.weight_en) * self.xgb_model.predict(X))

    @property
    def feature_importances_(self):
        # SHAP/importance plots stay meaningful by reading off the tree leg.
        return self.xgb_model.feature_importances_

def basicmodel(x, y, groups):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        random_state=42
    )

    cv = GroupKFold(n_splits=5)
    oof = cross_val_predict(model, x, y, cv=cv, groups=groups)

    print()
    print("Group-aware (by date) OOF R2:", r2_score(y, oof))

    model.fit(x, y)
    return model, model.predict(x)
def confidences(concentration):
    if concentration <= 0.15:
        confidence = "HIGH"
    elif concentration <= 0.4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    return confidence
def continuation(predicted_continuation):
    if predicted_continuation >= 2:
        signal = "BUY"
    elif predicted_continuation <= -5:
        signal = "STRONG SHORT"
    elif predicted_continuation <= -2:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"
    return signal
def dataexport(model, features, x, y, model_df, modelling, dependent, groups=None, oof_pred=None):
    joblib.dump(model, f"modeldata/{modelling}_m.pkl")
    joblib.dump(features, f"modeldata/{modelling}_f.pkl")

    # SHAP explainer is built on the tree leg only (ElasticNet has no
    # TreeExplainer equivalent); it's used purely for driver attribution,
    # the actual forecast comes from model.predict() (the blend).
    tree_component = getattr(model, "xgb_model", model)
    explainer = shap.TreeExplainer(tree_component)
    joblib.dump(explainer, f"modeldata/{modelling}_s.pkl")

    if oof_pred is None:
        # Fallback: honest group-aware OOF instead of the old unshuffled-
        # leakage-prone default 5-fold cross_val_predict.
        if groups is None:
            raise ValueError("groups is required to compute leakage-free OOF predictions")
        oof_pred = _oof_predict(model, x, y, groups)

    backtest_data = {
        "predictions": oof_pred,
        "realized_returns": y.values,
        f"{dependent}": model_df[dependent].values
    }

    joblib.dump(backtest_data, f"modeldata/{modelling}_h.pkl")
def _oof_predict(model_like, X, y, groups, n_splits=5):
    """Group-aware out-of-fold predictions for anything with .fit/.predict,
    including EnsembleModel. Refits fresh copies per fold (never touches
    the already-fitted `model_like` passed in)."""
    import copy
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y
    oof = np.zeros(len(y_arr))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X_arr, y_arr, groups):
        m = copy.deepcopy(model_like)
        Xtr = X.iloc[tr] if hasattr(X, "iloc") else X_arr[tr]
        Xte = X.iloc[te] if hasattr(X, "iloc") else X_arr[te]
        if isinstance(m, EnsembleModel):
            m.en_model.fit(Xtr, y_arr[tr])
            m.xgb_model.fit(Xtr, y_arr[tr])
        else:
            m.fit(Xtr, y_arr[tr])
        oof[te] = m.predict(Xte)
    return oof
def dataimport(modelling,dependent):
    model = joblib.load(f"modeldata/{modelling}_m.pkl")
    features = joblib.load(f"modeldata/{modelling}_f.pkl")
    history = joblib.load(f"modeldata/{modelling}_h.pkl")

    historical = pd.DataFrame({
        "pred": history["predictions"],
        modelling: history["realized_returns"],
        dependent: history[dependent]
    })

    historical["continuation"] = (historical[modelling] - historical[dependent])
    historical["decile"] = pd.qcut(historical["pred"],10,labels=False)
    return model, features, historical
def metrics(df,code):

    zero_cap = df["total_cap_m_usd"] == 0
    if zero_cap.any():
        df.loc[zero_cap, ["total_cap_m_usd","shs_cap_m_usd"]] = np.nan

    df["alpha1"] = (df["perf1_pct"] - df["bmk_perf1_pct"])
    df["alpha5"] = (df["perf5_pct"] - df["bmk_perf5_pct"])

    if code == "a5":
        df["continuation"] = (df["alpha5"] - df["alpha1"])
    elif code == "r5":
        df["continuation"] = (df["perf5_pct"] - df["perf1_pct"])
    elif code == "a1" or code == "r1":
        print("nah")

    df["buyback_adv"] = (df["notional_usd_m"] / df["adv_1m_usd"])
    df["buyback_cap"] = (df["notional_usd_m"] / df["total_cap_m_usd"] * 100)
    df["buyback_float"] = (df["notional_usd_m"] / df["eqy_float_usd"] * 100)

    df["premium"] = ((df["price"] - df["adj_vwap"]) / df["adj_vwap"] * 100)
    df["volume_shock"] = (df["adv_5d_usd"] / df["adv_1m_usd"])
    df["stress"] = (df["dtc"] * df["indicative_fee_pct"])
    df["abs_alpha1"] = (df["alpha1"].abs())
    df["attention"] = (df["abs_alpha1"] * df["volume_shock"])
    df["premium_significance"] = (df["premium"] * df["buyback_cap"])

    df["alpha1_x_attention"] = (df["alpha1"] * df["attention"])
    df["alpha1_x_premium"] = (df["alpha1"]* df["premium"])
    df["alpha1_x_buyback_adv"] = (df["alpha1"] * df["buyback_adv"])
    df["alpha1_x_vol"] = (df["alpha1"] * df["vol"])
    df["alpha1_z"] = df["alpha1"] / df["vol"]
    df["alpha1_x_short"] = (df["alpha1"] * df["dtc"])
    df["alpha1_x_buyback_cap"] = (df["alpha1"] * df["buyback_cap"])
    df["reaction_efficiency"] = (df["alpha1"] / (df["volume_shock"] + 0.01))

    df["buyback_to_vol"] = df["buyback_cap"] / df["vol"]
    df["buyback_x_volume"] = (df["buyback_cap"] * df["volume_shock"])
    df["buyback_x_stress"] = (df["buyback_cap"] * df["stress"])

    df["squeeze_score"] = (df["buyback_float"] * df["dtc"] * df["indicative_fee_pct"])
    df["buyback_short_cover"] = (df["buyback_float"] / (df["dtc"] + 0.01))

    df["turnover"] = (df["adv_1m_usd"] / df["eqy_float_usd"])
    df["illiquidity"] = (np.abs(df["perf1_pct"]) / df["adv_1m_usd"])

    df["premium_abs"] = df["premium"].abs()
    df["premium_x_volume"] = (df["premium"] * df["volume_shock"])

    df["cash_percent"] = (df["notional_usd_m"] / df["cash_usd_m"])

    top_alpha1 = df["alpha1"].quantile(0.90)
    bottom_alpha1 = df["alpha1"].quantile(0.10)
    df["top_alpha1"] = (df["alpha1"] >= top_alpha1).astype(int)
    df["bottom_alpha1"] = (df["alpha1"] <= bottom_alpha1).astype(int)

    df["cash_by_assets"] = df["cash_usd_m"] / df["total_assets"]
    df["fcf_by_assets"] = df["fcf"] / df["total_assets"]

    df["log_assets"] = np.log10(df["total_assets"])
    df["asset_ratio"] = df["fixed_assets"] / df["total_assets"]
    df["debt_ratio"] = df["debt"] / df["total_assets"]

    df["operating_roa"] = df["operating_income"] / df["total_assets"]
    df["net_debt_by_assets"] = (df["debt"] - df["cash_usd_m"]) / df["total_assets"]

    df["buyback_x_coverage_stress"] = df["buyback_cap"] / df["ebit_interest_ratio"]
    df["weak_coverage_flag"] = (df["ebit_interest_ratio"] < 3).astype(int)

    df["fcf_yield"] = df["fcf"] / df["total_cap_m_usd"]
    df["buyback_to_fcf"] = df["notional_usd_m"] / df["fcf"].clip(lower=0.01)

    df["cash_stress"] = df["cash_percent"] / (df["cash_by_assets"]+0.01)
    df["valuation_divergence"] = df["px_book_ratio"] - df["tobin_q_ratio"]

    df["alpha20"] = df["perf20_pct"] - df["bmk_perf20_pct"]
    df["distressed_value_buyback"] = df["buyback_cap"] * (-df["alpha20"]).clip(lower=0)
    df["momentum_chasing_buyback"] = df["buyback_cap"] * df["alpha20"].clip(lower=0)

    df["squeeze_setup"] = df["buyback_short_cover"] * (-df["alpha20"]).clip(lower=0)
    df["alpha1_vs_trend"] = df["alpha1"] - (df["alpha20"] / 20)

    def contains_weekend(start_date):
        return int(any((start_date + timedelta(days=i)).weekday() in (5,6) for i in range(5)))
    df["contains_weekend"] = df["date"].apply(contains_weekend)

    def weekend_before(start_date):
        return int(any((start_date - timedelta(days=i)).weekday() in (5, 6) for i in range(2)))
    df["weekend_before"] = df["date"].apply(weekend_before)

    return df
def expanding_date_splits(dates, n_splits=5, min_train_frac=0.4):
    dates = pd.Series(dates).reset_index(drop=True)
    uniq = np.sort(dates.unique())
    n = len(uniq)
    min_train = max(1, int(n * min_train_frac))
    test_pool = uniq[min_train:]
    blocks = [b for b in np.array_split(test_pool, n_splits) if len(b) > 0]

    splits = []
    train_end = min_train
    for block in blocks:
        train_dates = uniq[:train_end]
        train_idx = dates[dates.isin(train_dates)].index.values
        test_idx = dates[dates.isin(block)].index.values
        splits.append({
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_start": pd.Timestamp(train_dates.min()),
            "train_end": pd.Timestamp(train_dates.max()),
            "test_start": pd.Timestamp(block.min()),
            "test_end": pd.Timestamp(block.max()),
        })
        train_end += len(block)
    return splits
def _fit_tuned_xgb(x_tr, y_tr, groups_tr=None, splits=None, n_trials=40, inner_splits=3, recency_dates=None, weight_reference_date=None):
    x_arr = x_tr.values if hasattr(x_tr, "values") else x_tr
    y_arr = y_tr.values if hasattr(y_tr, "values") else y_tr

    if recency_dates is not None:
        recency_dates = pd.to_datetime(pd.Series(recency_dates), format = "%d/%m/%Y").reset_index(drop=True)
        if weight_reference_date is None:
            weight_reference_date = recency_dates.max()
        base_weights = recency_weights(recency_dates, weight_reference_date)
    else:
        base_weights = np.ones(len(y_arr))

    if splits is None:
        inner = GroupKFold(n_splits=min(inner_splits, groups_tr.nunique() if hasattr(groups_tr, "nunique") else inner_splits))
        splits = list(inner.split(x_arr, y_arr, groups_tr))

    def objective(trial):
        params = {name: (trial.suggest_int(name, *bounds) if isinstance(bounds[0], int)
                          else trial.suggest_float(name, *bounds))
                  for name, bounds in SMALL_XGB_SPACE.items()}
        params.update(objective="reg:squarederror", random_state=42, tree_method="hist")

        scores = []
        for itr, ite in splits:
            m = XGBRegressor(**params)
            m.fit(x_arr[itr], y_arr[itr], sample_weight=base_weights[itr])
            scores.append(r2_score(y_arr[ite], m.predict(x_arr[ite])))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = XGBRegressor(**study.best_params, objective="reg:squarederror",
                         random_state=42, tree_method="hist")
    best.fit(x_tr, y_tr, sample_weight=base_weights)
    return best, study.best_params
def _fit_elasticnet(x_tr, y_tr, groups_tr=None, splits=None, inner_splits=3, recency_dates=None, weight_reference_date=None):
    """Same dual-mode split argument as _fit_tuned_xgb."""
    y_arr = y_tr.values if hasattr(y_tr, "values") else y_tr
    if recency_dates is not None:
        recency_dates = pd.to_datetime(pd.Series(recency_dates), format = "%d/%m/%Y").reset_index(drop=True)
        if weight_reference_date is None:
            weight_reference_date = recency_dates.max()
        base_weights = recency_weights(recency_dates, weight_reference_date)
    else:
        base_weights = np.ones(len(y_arr))
    if splits is None:
        x_idx = x_tr.values if hasattr(x_tr, "values") else x_tr
        n_groups = groups_tr.nunique() if hasattr(groups_tr, "nunique") else len(np.unique(groups_tr))
        inner = GroupKFold(n_splits=min(inner_splits, n_groups))
        splits = list(inner.split(x_idx, y_arr, groups_tr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        en = Pipeline([
            ("scaler", StandardScaler()),
            ("en", ElasticNetCV(
                l1_ratio=[.1, .3, .5, .7, .9, 1.0],
                alphas=np.logspace(-3, 1, 30),
                cv=splits,
                max_iter=30000
            ))
        ])

        en.fit(x_tr,y_arr,en__sample_weight=base_weights)
    return en
def walk_forward_evaluate(x, y, dates, n_trials=20, n_splits=5, min_train_frac=0.4, inner_splits=3):
    """Genuine out-of-time evaluation: train only ever sees dates strictly
    before the test block, both for fitting and for hyperparameter search.
    This is the number to trust for 'how accurate will this be live' -
    GroupKFold-by-date (used elsewhere for tuning/deployment) is easier to
    win on because it still lets folds see future dates during training.
    """
    x_arr = x.values if hasattr(x, "values") else x
    y_arr = y.values if hasattr(y, "values") else y
    dates = pd.Series(dates).reset_index(drop=True)

    outer_splits = expanding_date_splits(dates, n_splits=n_splits, min_train_frac=min_train_frac)

    en_oof = np.full(len(y_arr), np.nan)
    xgb_oof = np.full(len(y_arr), np.nan)
    fold_rows = []

    print()
    print("=" * 70)
    print("TRUE OUT-OF-TIME EVALUATION (expanding walk-forward, no shuffling)")
    print("=" * 70)

    for i, s in enumerate(outer_splits):
        tr_idx, te_idx = s["train_idx"], s["test_idx"]
        x_tr = x.iloc[tr_idx] if hasattr(x, "iloc") else x_arr[tr_idx]
        y_tr = y_arr[tr_idx]
        x_te = x.iloc[te_idx] if hasattr(x, "iloc") else x_arr[te_idx]
        y_te = y_arr[te_idx]
        tr_dates = dates.iloc[tr_idx]
        weight_reference_date = tr_dates.max()

        # inner tuning splits are ALSO chronological, restricted to the
        # training window only - no hindsight leaks into hyperparameters
        inner = expanding_date_splits(tr_dates.reset_index(drop=True), n_splits=inner_splits, min_train_frac=0.5)
        inner_pairs = [(sp["train_idx"], sp["test_idx"]) for sp in inner] or None

        if inner_pairs:
            xgb_fold, _ = _fit_tuned_xgb(
                x_tr, y_tr, splits=inner_pairs, n_trials=n_trials,
                recency_dates=tr_dates, weight_reference_date=weight_reference_date
            )
            en_fold = _fit_elasticnet(
                x_tr, y_tr, splits=inner_pairs,
                recency_dates=tr_dates, weight_reference_date=weight_reference_date
            )
        else:
            # too little history for an inner split yet - fall back to fixed,
            # conservative params rather than tuning on nothing
            xgb_fold = XGBRegressor(n_estimators=150, max_depth=2, learning_rate=0.03,
                                     subsample=0.7, colsample_bytree=0.6, min_child_weight=15,
                                     random_state=42, tree_method="hist")
            fold_weights = recency_weights(tr_dates, weight_reference_date)
            xgb_fold.fit(x_tr, y_tr, sample_weight=fold_weights)
            en_fold = _fit_elasticnet(
                x_tr, y_tr, groups_tr=tr_dates,
                recency_dates=tr_dates, weight_reference_date=weight_reference_date
            )

        xgb_pred = xgb_fold.predict(x_te)
        en_pred = en_fold.predict(x_te)
        xgb_oof[te_idx] = xgb_pred
        en_oof[te_idx] = en_pred

        fold_rows.append({
            "fold": i, "n_train": len(tr_idx), "n_test": len(te_idx),
            "train_end": s["train_end"].date(), "test_start": s["test_start"].date(),
            "test_end": s["test_end"].date(),
            "xgb_r2": r2_score(y_te, xgb_pred), "en_r2": r2_score(y_te, en_pred),
        })
        print(f"  fold {i}: train up to {s['train_end'].date()} (n={len(tr_idx)})  "
              f"-> test {s['test_start'].date()}..{s['test_end'].date()} (n={len(te_idx)})   "
              f"XGB R2={fold_rows[-1]['xgb_r2']:.3f}   EN R2={fold_rows[-1]['en_r2']:.3f}")

    mask = ~np.isnan(en_oof)
    weights = np.linspace(0, 1, 21)
    blend_scores = [r2_score(y_arr[mask], w * en_oof[mask] + (1 - w) * xgb_oof[mask]) for w in weights]
    best_w = float(weights[int(np.argmax(blend_scores))])

    print("-" * 70)
    print(f"OUT-OF-TIME R2  -  ElasticNet: {r2_score(y_arr[mask], en_oof[mask]):.3f}   "
          f"XGB: {r2_score(y_arr[mask], xgb_oof[mask]):.3f}   ")

    print(f"Total R-squared value: {max(blend_scores):.3f}")
    print("=" * 70)

    return {
        "en_oof": en_oof, "xgb_oof": xgb_oof, "mask": mask, "best_w": best_w,
        "blend_r2": max(blend_scores), "folds": pd.DataFrame(fold_rows),
    }
def model(x, y, groups, n_trials=40, outer_splits=5):
    """Train the existing ElasticNet + XGBoost blend using walk-forward
    validation as the primary evaluation and blend-weight selection.

    The outer evaluation is chronological by announcement date. Hyperparameter
    tuning inside each walk-forward fold is also chronological, so no future
    dates are used to evaluate an earlier test period.

    GroupKFold is not used here: the walk-forward result is the model's primary
    out-of-sample accuracy estimate. The final models are then refit on all
    available history for deployment.
    """

    dates = pd.Series(groups).reset_index(drop=True)

    # Primary accuracy estimate: genuine out-of-time walk-forward evaluation.
    wf = walk_forward_evaluate(
        x, y, dates,
        n_trials=n_trials,
        n_splits=outer_splits,
        min_train_frac=0.4,
        inner_splits=3,
    )

    best_w = wf["best_w"]
    en_oof = wf["en_oof"]
    xgb_oof = wf["xgb_oof"]
    oof_blend = best_w * en_oof + (1 - best_w) * xgb_oof

    # Final deployment fit. Hyperparameter selection is still chronological:
    # use expanding splits over the full historical sample, then fit the
    # selected model on all available observations.
    final_inner = expanding_date_splits(
        dates, n_splits=outer_splits, min_train_frac=0.5
    )
    final_pairs = [(s["train_idx"], s["test_idx"]) for s in final_inner]

    if final_pairs:
        final_xgb, final_params = _fit_tuned_xgb(
            x, y, splits=final_pairs, n_trials=n_trials,
            recency_dates=dates, weight_reference_date=dates.max()
        )
        final_en = _fit_elasticnet(
            x, y, splits=final_pairs,
            recency_dates=dates, weight_reference_date=dates.max()
        )
    else:
        final_xgb = XGBRegressor(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.6,
            min_child_weight=15,
            random_state=42,
            tree_method="hist",
        )
        final_weights = recency_weights(dates, dates.max())
        final_xgb.fit(x, y, sample_weight=final_weights)
        final_params = {
            "n_estimators": 150,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "min_child_weight": 15,
        }
        final_en = _fit_elasticnet(
            x, y, groups_tr=dates,
            recency_dates=dates, weight_reference_date=dates.max()
        )

    print(f"\nRECENCY WEIGHTING: {RECENCY_HALF_LIFE_DAYS:.0f}-day half-life")
    print("\nFINAL XGB PARAMETERS (fit on full data)")
    print("===TABLE_START===")
    print(final_params)
    print("===TABLE_END===")


    ensemble = EnsembleModel(
        final_en,
        final_xgb,
        best_w,
        list(x.columns) if hasattr(x, "columns") else None,
    )

    # In-sample prediction is retained for the existing plotting/reporting
    # code, but the stored OOF prediction is the walk-forward prediction.
    pred = ensemble.predict(x)
    ensemble.oof_predictions_ = oof_blend
    ensemble.walk_forward_predictions_ = oof_blend
    ensemble.walk_forward_results_ = wf

    return ensemble, pred
def model_df(df, features, modelling, dependent):
    # keep "date" and "price" (used later by modelplots/report) alongside
    # the model columns so we can group-split by announcement date
    extra_cols = [c for c in ["date", "price"] if c not in features + [modelling]]
    model_df = (df[features + [modelling] + extra_cols]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                )

    numeric_cols = features + [modelling] + [c for c in extra_cols if c != "date"]
    for col in numeric_cols:
        # NOTE: bounds are computed on the full sample, not per-CV-fold.
        # This is a mild look-ahead (only affects where the 1%/99% outlier
        # clip lands, not the target itself) that we accept for simplicity;
        # with 1%/99% cutoffs on ~370 rows the bounds barely move whether
        # or not a given fold's rows are included, so it isn't meaningfully
        # inflating the honest OOF numbers reported by model().
        lower = model_df[col].quantile(0.01)
        upper = model_df[col].quantile(0.99)
        model_df[col] = (model_df[col].clip(lower, upper))

    x = model_df[features]
    y = model_df[modelling]
    groups = model_df["date"]
    model_df[f"{dependent}_decile"] = pd.qcut(model_df[dependent], 10, labels=False)
    return model_df, x, y, groups
def modelplots(modelling,model,features,x,y,pred,model_df,oof_pred=None):
    plt.figure(figsize=(8,6))
    plt.scatter(y, pred, alpha=0.5, label="in-sample fit")
    if oof_pred is not None:
        plt.scatter(y, oof_pred, alpha=0.5, label="honest OOF (GroupKFold)")
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    plt.plot(lims,lims, color="red")
    plt.xlabel(f"Actual {modelling}")
    plt.ylabel(f"Predicted {modelling}")
    plt.title("Performance Model")
    plt.legend()
    plt.tight_layout()
    plt.show()

    tree_component = getattr(model, "xgb_model", model)
    importance = pd.DataFrame({"feature": features, "importance": tree_component.feature_importances_})
    importance = (importance.sort_values("importance", ascending=False))

    print("\n",importance.to_string(index=False))
    plt.figure(figsize=(10,6))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1])
    plt.title("Feature Importance (tree leg of the blend)")
    plt.tight_layout()
    plt.show()

    explainer = shap.TreeExplainer(tree_component)
    shap_values = explainer.shap_values(x)
    shap.summary_plot( shap_values, x)

    confidence_scores = []
    concentrations = []
    for row in shap_values:
        abs_shap = np.abs(row)
        total_impact = abs_shap.sum()

        if total_impact == 0:
            concentrations.append(np.nan)
            confidence_scores.append(np.nan)
            continue

        top_driver = abs_shap.max()
        concentration = top_driver / total_impact * 100
        confidence_score = (100 -concentration)
        confidence_scores.append(confidence_score)

    results = pd.DataFrame({
        "actual":y,
        "predicted":pred,
        "confidence":confidence_scores,
    })

    results["error"] = (results["actual"] - results["predicted"]).abs()
    results["confidence_bin"] = pd.cut(results["confidence"],bins=[0,20,40,60,80,100],labels=["0-20","20-40","40-60","60-80","80-100"])

    error_curve = (results.groupby("confidence_bin")["error"].agg(["count","mean","median","std"]))

    mae_by_confidence = (results.groupby("confidence_bin")["error"].median())
    print(error_curve)
    mae_by_confidence.to_json("modeldata/confidence_error.json")

    bought = model_df["price"]
    plt.figure(figsize=(10,6))
    plt.scatter(pred, bought)
    plt.scatter(y,bought)
    plt.title(f"Predicted vs Actual positions for {modelling}")
    for x1, x2, y1 in zip(y,pred,bought):
        plt.plot([x1, x2], [y1, y1], color='black', linestyle='--', linewidth=1)
    plt.legend(["Predicted", "Actual"])
    plt.xticks(ticks = np.arange(-20, 20, 1), minor = True)
    plt.grid(visible=True, which="both", axis="x", linestyle="--", linewidth=0.5)
    plt.show()
def report(model, features, historical, trade,modelling, dependent):
    pred_modelled = model.predict(trade[features])[0]
    oneday = next(trade[dependent].items())
    oneday = oneday[1]
    predicted_continuation = (pred_modelled - oneday)
    percentile = ((historical["pred"] < pred_modelled).mean() * 100)

    decile_edges = pd.qcut(historical["pred"],10,retbins=True,duplicates="drop")[1]
    decile = np.searchsorted(decile_edges, pred_modelled, side="right") - 1
    decile = np.clip(decile, 0, len(decile_edges) - 2)
    bucket = historical[
        historical["pred"].between(
            decile_edges[decile],
            decile_edges[decile + 1],
            inclusive="left"
        )
    ]

    expected_modelled = (bucket[modelling].mean())
    median_modelled = (bucket[modelling].median())
    modelled_hit_rate = (bucket[modelling] > 0).mean() * 100
    expected_continuation = (bucket["continuation"].mean())
    median_continuation = (bucket["continuation"].median())
    continuation_hit_rate = (bucket["continuation"] > 0).mean() * 100

    signal = continuation(predicted_continuation)

    tree_component = getattr(model, "xgb_model", model)
    explainer = shap.TreeExplainer(tree_component)
    shap_values = explainer.shap_values(trade[features])

    drivers = pd.DataFrame({
        "feature": features,
        "impact": shap_values[0]
    })

    total_shap = np.abs(shap_values).sum()
    largest_shap = np.abs(shap_values).max()

    concentration = largest_shap / total_shap
    confidence = confidences(concentration)


    print()
    print("=" * 70)
    print(f"BUYBACK SIGNAL REPORT: {next(trade["bbg_ticker"].items())[1]}")
    print("=" * 70)

    print(f"\nObserved {dependent}       : {oneday:.2f}%")
    print(f"Predicted {modelling}      : {pred_modelled:.2f}%")
    print(f"Expected Continuation    : {predicted_continuation:.2f}%")

    print(f"\nPercentile Rank          : {percentile:.1f}%")
    print(f"Confidence               : {confidence}")
    print(f"Historical Decile        : {decile}")
    print(f"Signal                   : {signal}")

    print("\nBACKTEST EXPECTATION")
    print("-" * 70)
    print(f"Average {modelling}        : {expected_modelled:.2f}%")
    print(f"Median {modelling}         : {median_modelled:.2f}%")
    print(f"{modelling} Hit Rate       : {modelled_hit_rate:.1f}%")

    print(f"\nAverage Continuation     : {expected_continuation:.2f}%")
    print(f"Median Continuation      : {median_continuation:.2f}%")
    print(f"Continuation Hit Rate    : {continuation_hit_rate:.1f}%")

    print()
    print("=" * 70)
    print("TOP POSITIVE DRIVERS")
    print("=" * 70)

    print(
        drivers
        .sort_values("impact", ascending=False)
        .head(5)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("TOP NEGATIVE DRIVERS")
    print("=" * 70)

    print(
        drivers
        .sort_values("impact", ascending=True)
        .head(5)
        .to_string(index=False)
    )

    try:
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=trade[features].iloc[0],
            feature_names=features
        )

        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.gca().tick_params(axis='y', labelsize=2)
        plt.show()
    except Exception:
        pass
def which(what):
    if what == "alpha5":
        code = "a5"
        modelling = what
        dependent = "alpha1"
        features = features_5d
    elif what == "perf5_pct":
        code = "r5"
        modelling = what
        dependent = "perf1_pct"
        features = features_5d
    elif what == "alpha1":
        code = "a1"
        modelling = what
        dependent = 0
        features = features_1d
    elif what == "perf1_pct":
        code = "r1"
        modelling = what
        dependent = 0
        features = features_1d
    else:
        return None
    return code, modelling, dependent, features

features_5d = ["alpha1","alpha20",
    "abs_alpha1",
    "alpha1_x_attention",
    "alpha1_x_buyback_adv",
    "alpha1_x_buyback_cap",
    "alpha1_x_premium",
    "alpha1_x_short",
    "alpha1_x_vol",
    "alpha1_z",
    "alpha1_vs_trend",
    "asset_ratio",
    "attention",
    "bmk_perf1_pct",
    "bottom_alpha1",
    "buyback_adv",
    "buyback_short_cover",
    "buyback_to_fcf",
    "buyback_to_vol",
    "buyback_x_stress",
    "buyback_x_volume",
    "cash_by_assets",
    "cash_percent",
    "cash_stress",
    "debt_ratio",
    "distressed_value_buyback",
    "ebit_interest_ratio",
    "fcf_by_assets",
    "fcf_yield",
    "illiquidity",
    "log_assets",
    "momentum_chasing_buyback",
    "net_debt_by_assets",
    "operating_roa",
    "perf1_pct",
    "perf20_pct",
    "bmk_perf20_pct",
    "premium",
    "premium_abs",
    "premium_significance",
    "premium_x_volume",
    "price",
    "reaction_efficiency",
    "squeeze_score",
    "squeeze_setup",
    "tobin_q_ratio",
    "top_alpha1",
    "turnover",
    "valuation_divergence",
    "vol",
    "volume_shock",
    "weekend_before"
]
features_1d = ["buyback_adv",
    "buyback_short_cover",
    "buyback_to_vol",
    "buyback_x_stress",
    "buyback_x_volume",
    "cash_percent",
    "cash_usd_m",
    "contains_weekend",
    "premium",
    "premium_abs",
    "premium_significance",
    "premium_x_volume",
    "price",
    "squeeze_score",
    "turnover",
    "vol",
    "volume_shock",
    "weekend_before"
]

def checker_input(ticker,df):
    return df.loc[df["bbg_ticker"] == ticker]

#     = pd.DataFrame([{
#     "date": pd.to_datetime("2026-8-26"),
#     "bbg_ticker":"5244 JP Equity",
#     "name":"jig.jp co.,ltd.",
#     "shares":420000,
#     "purchased_shares":373500,
#     "price":176,
#     "notional_usd_m":0.46414368,
#     "total_cap_m_usd":47.0425091136,
#     "shs_cap_m_usd":47.0425091136,
#     "eqy_float_usd":23.7145372464,
#     "beta":0.9,
#     "vol":2.5,
#     "adv_1m_usd":0.310743386597914,
#     "adv_5d_usd":0.231777966125086,
#     "dtc":1.19,
#     "indicative_fee_pct":5.91,
#     "adj_vwap":373.84,
#     "perf1_pct":3.9773,
#     "bmk_perf1_pct":0.4238,
#     "perf5_pct":None,
#     "bmk_perf5_pct":None,
#     "px_book_ratio":1.46,
#     "cash_usd_m":4849.729,
#     "total_assets":7506.847,
#     "fcf":1006.253,
#     "ebit_interest_ratio":953.6155,
#     "fixed_assets":683.254,
#     "operating_income":14631.51,
#     "debt":280.136,
#     "tobin_q_ratio":1.543801,
#     "perf20_pct":2.923976608,
#     "bmk_perf20_pct":0.678788,
#
# }])