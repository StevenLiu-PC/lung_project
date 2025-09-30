# encoding: utf-8
# eol: LF
# summary: feat(day10): 多模型 K-fold 比較、最佳模型重訓與導出

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

# ===== 0) 路徑 & 參數（先讀 project_config，讀不到就 fallback）=====
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
RESULTS_CSV = ARTIFACTS_DIR / "cv_results_day10.csv"# 交叉驗證結果 (CSV 檔) -> 存模型在不同參數下的表現數值
BEST_MODEL_PKL = ARTIFACTS_DIR / "best_model_day10.pkl" # 最佳模型 (pickle 格式) -> 存訓練完成的模型物件
REPORT_TXT    = ARTIFACTS_DIR / "day10_report.txt" #  報告檔 (純文字) -> 紀錄最佳模型的表現與相關參數

# ★ 讓相對路徑更穩：以專案根 (lung_project) 為基準
ROOT = Path(__file__).resolve().parent.parent  # ★ lung_project
if not Path(INPUT_CSV_PATH).is_absolute():     # ★ 若是相對路徑，補上 ROOT
    # ★ 你 Day4 主輸出實際是 survey_lung_day4.csv；若 project_config 沒成功匯入，這裡校正
    maybe = ROOT / "data_lung" / "processed" / "survey_lung_day4.csv"  # ★
    INPUT_CSV_PATH = maybe if maybe.exists() else ROOT / INPUT_CSV_PATH # ★

# ===== 1) 共用工具（延續 Day 9 命名與寫法）=====
def read_csv_robust(path):
    """穩健讀檔：先讓 pandas 自動猜分隔；不行再試常見 sep/encoding。"""
    p = Path(path)  # ★ 統一路徑物件
    if not p.exists():
        raise FileNotFoundError(f"[Day10] 找不到檔案：{p}")  # ★
    if p.stat().st_size == 0:
        raise ValueError(f"[Day10] 檔案是空的：{p}")       # ★

    # 先試 sep=None（需 engine='python'）
    try:
        # ★ 優先用 utf-8-sig（處理 BOM），失敗再退回
        df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")  # ★
        if df.shape[0] > 0 and df.shape[1] > 1:
            # ★ 清掉 BOM 欄名（\ufeff），以及意外索引欄
            df.columns = [c.lstrip("\ufeff") for c in df.columns]             # ★
            if "Unnamed: 0" in df.columns:                                    # ★
                df = df.drop(columns=["Unnamed: 0"])                           # ★
            return df
    except Exception:
        pass

    # fallback：常見分隔與編碼（把 utf-8-sig 放最前面）
    for enc in ["utf-8-sig", "utf-8", "cp950"]:           # ★
        for sep in [None, ",", ";", "\t", "|"]:            # ★ None 也讓它再試一次
            try:
                df = pd.read_csv(p, sep=sep, encoding=enc, engine="python")   # ★
                if df.shape[0] > 0 and df.shape[1] > 1:
                    df.columns = [c.lstrip("\ufeff") for c in df.columns]     # ★
                    if "Unnamed: 0" in df.columns:                            # ★
                        df = df.drop(columns=["Unnamed: 0"])                   # ★
                    print(f"[Day10] read_csv_robust: sep={repr(sep)}, encoding='{enc}'")  # ★
                    return df
            except Exception:
                continue

    raise ValueError("[Day10] CSV 讀取失敗或內容無效，請檢查 Day4 輸出/分隔符/編碼（建議 Day4 以 utf-8-sig 寫出）。")  # ★

def check_target_column(dataframe: pd.DataFrame, target_column: str) -> None:
    """確認目標欄位是否存在且至少兩類。（與 Day9 一致）"""
    if dataframe.empty:
        raise ValueError("[Day10] 讀入資料為空(0 列）。")
    if target_column not in dataframe.columns:
        raise ValueError(f"[Day10] 找不到目標欄：{target_column}；實際欄位預覽：{list(dataframe.columns)[:10]}")
    nunique = dataframe[target_column].nunique(dropna=True)
    if nunique < 2:
        vc = dataframe[target_column].value_counts(dropna=False).to_dict()
        raise ValueError(f"[Day10] 目標欄 `{target_column}` 類別不足。分佈：{vc}")

def prepare_features_and_target(df: pd.DataFrame, target_column: str):
    """拆分特徵與標籤，並處理 one-hot / 缺值。（與 Day9 一致）"""
    categorical_columns = [c for c in df.select_dtypes(include="object").columns if c != target_column]
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dummy_na=True)
        print(f"[Day10] One-hot 編碼欄位：{categorical_columns}")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y

# ===== 2) 主流程：多模型＋交叉驗證 → 挑最佳 → 全資料重訓 → 存檔 =====
def run_day10(
    input_csv_path: str | Path = INPUT_CSV_PATH,
    target_column: str = TARGET_COLUMN_NAME,
    cv_folds: int = 5,
    random_state: int = 42,
    select_metric: str = "roc_auc",  # 可選：'roc_auc' / 'f1' / 'accuracy'
):
    """Day10:讀檔 → 前處理 → 多模型交叉驗證 → 挑最佳 → 全資料重訓 → 存檔"""
    # --- 2.1 讀檔 & 檢查（維持你 Day1~9 的 print 格式）---
    df = read_csv_robust(input_csv_path)
    print(f"[Day10] 讀入：{input_csv_path}, shape={df.shape}")
    # ★ 先把欄名去除 BOM，以防 project_config 的 TARGET_COL 沒對齊
    df.columns = [c.lstrip("\ufeff") for c in df.columns]  # ★
    check_target_column(df, target_column)

    # --- 2.2 標籤標準化成數字 0/1（兼容 YES/NO、Y/N、TRUE/FALSE、是/否、以及 1/2）---
    if df[target_column].dtype == object:
        mapper = {"YES": 1, "NO": 0, "Y": 1, "N": 0, "TRUE": 1, "FALSE": 0, "是": 1, "否": 0}
        df[target_column] = (
            df[target_column].astype(str).str.strip().str.upper().map(mapper)
        )
        # ★ 如果仍有無法識別值，先嘗試把 '1'/'2' 轉成數字
        if df[target_column].isna().any():                                # ★
            try:                                                          # ★
                tmp = pd.to_numeric(df[target_column], errors="coerce")   # ★
                df[target_column] = tmp                                   # ★
            except Exception:
                pass
    # ★ 若目標欄是 {1,2}（UCI 原始：1=Yes, 2=No），轉成 {1,0}
    vals = set(pd.Series(df[target_column]).dropna().unique().tolist())   # ★
    if vals <= {0,1,2}:                                                   # ★
        df[target_column] = df[target_column].replace({2: 0, 1: 1})       # ★

    # 確保為整數 0/1
    df[target_column] = df[target_column].astype(int)

    # --- 2.3 特徵與標籤 ---
    X, y = prepare_features_and_target(df, target_column)

    # --- 2.4 交叉驗證設定（分層 K 折，保持類別比例）---
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # --- 2.5 模型列表（尺度敏感的前面加 StandardScaler；命名與你日常一致）---
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

    # --- 2.6 多指標 scoring（和 Day9 的語感一致）---
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    # --- 2.7 逐一模型做交叉驗證，彙整成表格 ---
    rows = []
    for name, pipe in models.items():
        print(f"[Day10] 交叉驗證中：{name}")
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
    print(f"[Day10] 交叉驗證完成，結果已輸出：{RESULTS_CSV}")

    # --- 2.8 挑選最佳模型（依 select_metric 排序第一）---
    best_row = df_results.iloc[0]
    best_name = best_row["Model"]
    print(f"[Day10] 最佳模型：{best_name} | {select_metric}={best_row[f'{select_metric}_mean']:.4f}")

    # --- 2.9 最佳模型在「全資料」上重訓，存成 Day10 產物 ---
    best_pipe = models[best_name]
    best_pipe.fit(X, y)
    joblib.dump(best_pipe, BEST_MODEL_PKL)
    print(f"[Day10] 已存最佳模型至：{BEST_MODEL_PKL}")

    # --- 2.10 簡要文字報告（延續 Day9 報告風格）---
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

    print("[Day10] ✅ 完成，輸出：")
    print(f"- {RESULTS_CSV}")
    print(f"- {BEST_MODEL_PKL}")
    print(f"- {REPORT_TXT}")

    return ARTIFACTS_DIR

# === 快速診斷：檔案是否真的存在、有內容、分隔符可能是什麼 ===
p = Path(INPUT_CSV_PATH )
print("[debug] resolved path:", p.resolve())
print("[debug] exists:", p.exists())
print("[debug] size(bytes):", p.stat().st_size if p.exists() else -1)

if __name__ == "__main__":
    run_day10()
