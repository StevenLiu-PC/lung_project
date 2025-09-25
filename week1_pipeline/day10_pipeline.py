# Day10 pipeline: 多模型交叉驗證（LR/DT/RF/SVM），依 roc_auc 選最佳並重訓；輸出報告與結果表
# 註解：僅新增說明，不影響程式邏輯

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ===== Sklearn =====
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ===== 0) 頝臬? & ?嚗?霈 project_config嚗?銝撠?fallback嚗?====
try:
    from project_config import OUTPUT_CSV_DAY4 as INPUT_CSV_PATH
except Exception:
    INPUT_CSV_PATH = Path("data_lung/processed/day4_cleaned.csv")

try:
    from project_config import TARGET_COL as TARGET_COLUMN_NAME
except Exception:
    TARGET_COLUMN_NAME = "LUNG_CANCER"

ARTIFACTS_DIR = Path("data_lung/artifacts/day10") 
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = ARTIFACTS_DIR / "cv_results_day10.csv"# 鈭文?撽?蝯? (CSV 瑼? -> 摮芋?銝??銝?銵函?詨?
BEST_MODEL_PKL = ARTIFACTS_DIR / "best_model_day10.pkl" # ?雿單芋??(pickle ?澆?) -> 摮?蝺游???璅∪??拐辣
REPORT_TXT    = ARTIFACTS_DIR / "day10_report.txt" #  ?勗?瑼?(蝝?摮? -> 蝝??雿單芋??銵函?????

# ??霈撠楝敺蝛抬?隞亙?獢 (lung_project) ?箏皞?
ROOT = Path(__file__).resolve().parent.parent  # ??lung_project
if not Path(INPUT_CSV_PATH).is_absolute():     # ???交?詨?頝臬?嚗?銝?ROOT
    # ??雿?Day4 銝餉撓?箏祕? survey_lung_day4.csv嚗 project_config 瘝???伐??ㄐ?⊥迤
    maybe = ROOT / "data_lung" / "processed" / "survey_lung_day4.csv"  # ??
    INPUT_CSV_PATH = maybe if maybe.exists() else ROOT / INPUT_CSV_PATH # ??

# ===== 1) ?梁撌亙嚗辣蝥?Day 9 ?賢??神瘜?=====
def read_csv_robust(path):
    """蝛拙霈瑼??? pandas ?芸?????銝??岫撣貉? sep/encoding??""
    p = Path(path)  # ??蝯曹?頝臬??拐辣
    if not p.exists():
        raise FileNotFoundError(f"[Day10] ?曆??唳?獢?{p}")  # ??
    if p.stat().st_size == 0:
        raise ValueError(f"[Day10] 瑼??舐征??{p}")       # ??

    # ?岫 sep=None嚗? engine='python'嚗?
    try:
        # ???芸???utf-8-sig嚗???BOM嚗?憭望????
        df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")  # ??
        if df.shape[0] > 0 and df.shape[1] > 1:
            # ??皜? BOM 甈?嚗ufeff嚗?隞亙???蝝Ｗ?甈?
            df.columns = [c.lstrip("\ufeff") for c in df.columns]             # ??
            if "Unnamed: 0" in df.columns:                                    # ??
                df = df.drop(columns=["Unnamed: 0"])                           # ??
            return df
    except Exception:
        pass

    # fallback嚗虜閬???蝺函Ⅳ嚗? utf-8-sig ?暹??嚗?
    for enc in ["utf-8-sig", "utf-8", "cp950"]:           # ??
        for sep in [None, ",", ";", "\t", "|"]:            # ??None 銋?摰?閰虫?甈?
            try:
                df = pd.read_csv(p, sep=sep, encoding=enc, engine="python")   # ??
                if df.shape[0] > 0 and df.shape[1] > 1:
                    df.columns = [c.lstrip("\ufeff") for c in df.columns]     # ??
                    if "Unnamed: 0" in df.columns:                            # ??
                        df = df.drop(columns=["Unnamed: 0"])                   # ??
                    print(f"[Day10] read_csv_robust: sep={repr(sep)}, encoding='{enc}'")  # ??
                    return df
            except Exception:
                continue

    raise ValueError("[Day10] CSV 霈?仃???批捆?⊥?嚗?瑼Ｘ Day4 頛詨/??蝚?蝺函Ⅳ嚗遣霅?Day4 隞?utf-8-sig 撖怠嚗?)  # ??

def check_target_column(dataframe: pd.DataFrame, target_column: str) -> None:
    """蝣箄??格?甈??臬摮銝撠憿???Day9 銝?湛?"""
    if dataframe.empty:
        raise ValueError("[Day10] 霈?亥??蝛?0 ????)
    if target_column not in dataframe.columns:
        raise ValueError(f"[Day10] ?曆??啁璅?嚗target_column}嚗祕??雿?閬踝?{list(dataframe.columns)[:10]}")
    nunique = dataframe[target_column].nunique(dropna=True)
    if nunique < 2:
        vc = dataframe[target_column].value_counts(dropna=False).to_dict()
        raise ValueError(f"[Day10] ?格?甈?`{target_column}` 憿銝雲??雿?{vc}")

def prepare_features_and_target(df: pd.DataFrame, target_column: str):
    """???孵噩??蝐歹?銝西???one-hot / 蝻箏潦???Day9 銝?湛?"""
    categorical_columns = [c for c in df.select_dtypes(include="object").columns if c != target_column]
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dummy_na=True)
        print(f"[Day10] One-hot 蝺函Ⅳ甈?嚗categorical_columns}")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y

# ===== 2) 銝餅?蝔?憭芋??鈭文?撽? ????雿????刻???閮???摮? =====
def run_day10(
    input_csv_path: str | Path = INPUT_CSV_PATH,
    target_column: str = TARGET_COLUMN_NAME,
    cv_folds: int = 5,
    random_state: int = 42,
    select_metric: str = "roc_auc",  # ?舫嚗?roc_auc' / 'f1' / 'accuracy'
):
    """Day10:霈瑼?????????憭芋?漱??霅?????雿????刻???閮???摮?"""
    # --- 2.1 霈瑼?& 瑼Ｘ嚗雁?? Day1~9 ??print ?澆?嚗?--
    df = read_csv_robust(input_csv_path)
    print(f"[Day10] 霈?伐?{input_csv_path}, shape={df.shape}")
    # ????甈??駁 BOM嚗誑??project_config ??TARGET_COL 瘝?朣?
    df.columns = [c.lstrip("\ufeff") for c in df.columns]  # ??
    check_target_column(df, target_column)

    # --- 2.2 璅惜璅????詨? 0/1嚗摰?YES/NO?/N?RUE/FALSE?/?艾誑??1/2嚗?--
    if df[target_column].dtype == object:
        mapper = {"YES": 1, "NO": 0, "Y": 1, "N": 0, "TRUE": 1, "FALSE": 0, "??: 1, "??: 0}
        df[target_column] = (
            df[target_column].astype(str).str.strip().str.upper().map(mapper)
        )
        # ??憒?隞??⊥?霅?潘???閰行? '1'/'2' 頧??詨?
        if df[target_column].isna().any():                                # ??
            try:                                                          # ??
                tmp = pd.to_numeric(df[target_column], errors="coerce")   # ??
                df[target_column] = tmp                                   # ??
            except Exception:
                pass
    # ???亦璅???{1,2}嚗CI ??嚗?=Yes, 2=No嚗?頧? {1,0}
    vals = set(pd.Series(df[target_column]).dropna().unique().tolist())   # ??
    if vals <= {0,1,2}:                                                   # ??
        df[target_column] = df[target_column].replace({2: 0, 1: 1})       # ??

    # 蝣箔??箸??0/1
    df[target_column] = df[target_column].astype(int)

    # --- 2.3 ?孵噩??蝐?---
    X, y = prepare_features_and_target(df, target_column)

    # --- 2.4 鈭文?撽?閮剖?嚗?撅?K ??靽?憿瘥?嚗?--
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # --- 2.5 璅∪??”嚗偕摨行??????StandardScaler嚗??雿撣訾??湛?---
    models: dict[str, Pipeline] = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        "DecisionTree": Pipeline([
            ("clf", DecisionTreeClassifier(random_state=random_state))
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=random_state, n_jobs=-1
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", SVC(probability=True, random_state=random_state))
        ]),
    }

    # --- 2.6 憭?璅?scoring嚗? Day9 ?????湛?---
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    # --- 2.7 ??璅∪??漱??霅?敶?”??---
    rows = []
    for name, pipe in models.items():
        print(f"[Day10] 鈭文?撽?銝哨?{name}")
        cv_out = cross_validate(
            pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
        )
        rows.append({
            "Model":          name,
            "accuracy_mean":  np.mean(cv_out["test_accuracy"]),
            "accuracy_std":   np.std(cv_out["test_accuracy"]),
            "precision_mean": np.mean(cv_out["test_precision"]),
            "recall_mean":    np.mean(cv_out["test_recall"]),
            "f1_mean":        np.mean(cv_out["test_f1"]),
            "roc_auc_mean":   np.mean(cv_out["test_roc_auc"]),
        })

    df_results = pd.DataFrame(rows).sort_values(by=f"{select_metric}_mean", ascending=False)
    df_results.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"[Day10] 鈭文?撽?摰?嚗??歇頛詨嚗RESULTS_CSV}")

    # --- 2.8 ??雿單芋??靘?select_metric ??蝚砌?嚗?--
    best_row = df_results.iloc[0]
    best_name = best_row["Model"]
    print(f"[Day10] ?雿單芋??{best_name} | {select_metric}={best_row[f'{select_metric}_mean']:.4f}")

    # --- 2.9 ?雿單芋??鞈?????嚗???Day10 ?Ｙ ---
    best_pipe = models[best_name]
    best_pipe.fit(X, y)
    joblib.dump(best_pipe, BEST_MODEL_PKL)
    print(f"[Day10] 撌脣??雿單芋?嚗BEST_MODEL_PKL}")

    # --- 2.10 蝪∟????勗?嚗辣蝥?Day9 ?勗?憸冽嚗?--
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Day10 Cross-Validation Model Comparison\n")
        f.write(f"Data   : {input_csv_path}\n")
        f.write(f"Target : {target_column}\n")
        f.write(f"CV     : {cv_folds}-fold StratifiedKFold\n")
        f.write(f"Select : {select_metric}\n\n")
        f.write("=== Leaderboard (sorted) ===\n")
        f.write(df_results.to_string(index=False))
        f.write("\n\n")
        f.write(f"Best model: {best_name}\n")
        f.write(f"{select_metric}_mean: {best_row[f'{select_metric}_mean']:.4f}\n")

    print("[Day10] ??摰?嚗撓?綽?")
    print(f"- {RESULTS_CSV}")
    print(f"- {BEST_MODEL_PKL}")
    print(f"- {REPORT_TXT}")

    return ARTIFACTS_DIR

# === 敹恍那?瘀?瑼??臬??摮???批捆???泵?航?臭?暻?===
p = Path(INPUT_CSV_PATH )
print("[debug] resolved path:", p.resolve())
print("[debug] exists:", p.exists())
print("[debug] size(bytes):", p.stat().st_size if p.exists() else -1)

if __name__ == "__main__":
    run_day10()
