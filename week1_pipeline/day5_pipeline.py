# --- 確保能 import 專案根的 project_config.py ---
import sys
from pathlib import Path

# week1_pipeline/xxx_pipeline.py → 專案根 (lung_project)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------

"""
Day 5: Baseline 建模骨架
- 讀取 Day4 清理後資料 (day4_cleaned.csv)
- 確認/補回目標欄
- 類別欄 one-hot（排除目標欄）
- 訓練/測試切分
- 兩個 baseline: Logistic Regression / GaussianNB
- 指標與圖表輸出：metrics.csv、classification_report.txt、ROC、混淆矩陣、(logreg) 係數表
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 中文字體（可選）
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# === 路徑設定（優先吃 project_config，否則用預設） ===
try:
    from project_config import OUTPUT_CSV_DAY4 as INPUT_CSV  # day4_cleaned.csv
except Exception:
    INPUT_CSV = Path("data_lung/processed/day4_cleaned.csv")

# 目標欄名稱（若有在 project_config 指定 TARGET_COL 會優先採用）
try:
    from project_config import TARGET_COL as DEFAULT_TARGET
except Exception:
    DEFAULT_TARGET = "LUNG_CANCER"

ARTIFACTS = Path("artifacts_day5")
METRICS_CSV = ARTIFACTS / "metrics.csv"
CLF_REPORT_TXT = ARTIFACTS / "classification_report.txt"
ROC_PNG = ARTIFACTS / "roc_curve.png"
CM_DIR = ARTIFACTS / "confusion_matrices"
COEF_CSV = ARTIFACTS / "logreg_coefficients.csv"

# --------------------------
# 小工具：檢查目標欄
# --------------------------
def _check_target(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"[Day5] 目標欄 `{target}` 不存在於資料中。")
    nuniq = df[target].nunique(dropna=True)
    if nuniq < 2:
        raise ValueError(f"[Day5] 目標欄 `{target}` 只有 {nuniq} 個類別，無法進行分類。")

# --------------------------
# 小工具：若 Day4 沒有目標欄，嘗試自 Day3 補回
# --------------------------
def _maybe_restore_target_from_day3(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target in df.columns:
        return df
    try:
        day3 = pd.read_csv("data_lung/processed/day3_features.csv", usecols=[target])
        if len(day3) == len(df):
            df[target] = day3[target].values
            print(f"[Day5] 注意：`{target}` 不在 Day4 輸出，已從 Day3 補回。")
        else:
            print("[Day5] Day3 與 Day4 筆數不同，無法補回目標欄。")
    except Exception as e:
        print(f"[Day5] 無法從 Day3 補回目標欄 `{target}`：{e}")
    return df

# --------------------------
# 小工具：特徵前處理（排除目標欄）
# --------------------------
def _prepare_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """確保 X 全為數值。若仍有 object 欄，使用 pandas.get_dummies 轉為 one-hot（排除目標欄）。"""
    # 排除目標欄，避免把目標欄 one-hot 掉
    obj_cols = [c for c in df.select_dtypes(include="object").columns.tolist() if c != target]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True, dummy_na=True)
        print(f"[Day5] 已 one-hot 編碼欄位：{obj_cols}")

    # 取出 y 後，再對 X 做安全處理（±inf→NaN→0）
    y = df[target]
    X = df.drop(columns=[target])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y

# --------------------------
# 小工具：簡易 train/test split（避免強依賴 sklearn）
# --------------------------
def _train_test_split(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(idx)
    cut = int((1 - test_size) * n)
    tr_idx, te_idx = idx[:cut], idx[cut:]
    return X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]

# --------------------------
# 小工具：畫 ROC（多模型）
# --------------------------
def _plot_roc(y_test, y_proba_dict: dict[str, np.ndarray], out_path: Path):
    plt.figure()
    try:
        from sklearn.metrics import roc_curve, auc as _auc
        plotted = False
        for name, prob in y_proba_dict.items():
            if prob is None:
                continue
            fpr, tpr, _ = roc_curve(y_test, prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={_auc(fpr, tpr):.3f})")
            plotted = True
        if plotted:
            plt.plot([0, 1], [0, 1], "--")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (Day5 Baselines)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
        else:
            print("[Day5] 無可用機率輸出，略過 ROC 繪圖。")
    finally:
        plt.close()

# --------------------------
# 小工具：畫混淆矩陣
# --------------------------
def _plot_confusion_matrix(y_test, y_pred, out_path: Path):
    from sklearn.metrics import confusion_matrix
    import itertools
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks); plt.yticks(tick_marks, tick_marks)
    # 標註數字
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

# --------------------------
# 主流程
# --------------------------
def run_day5(
    input_csv: str | Path = INPUT_CSV,
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """執行 Day 5 baseline：Logistic / Naive Bayes，輸出指標與圖表。"""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    print(f"[Day5] 讀入：{input_csv} shape={df.shape}")

    # 若 Day4 沒有目標欄，先嘗試從 Day3 補回
    df = _maybe_restore_target_from_day3(df, target)

    # 若目標欄仍不存在/或只有單一類別，直接報錯以提醒修資料
    _check_target(df, target)

    # 若目標欄仍為字串（極少見，Day4 通常已數值化），嘗試轉 0/1
    if df[target].dtype == object:
        mapper = {"YES":1,"NO":0,"Y":1,"N":0,"TRUE":1,"FALSE":0,"是":1,"否":0,"1":1,"0":0,"1.0":1,"0.0":0}
        df[target] = df[target].astype(str).str.strip().str.upper().map(mapper)

    X, y = _prepare_features(df, target)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, test_size, random_state)

    # 載入 sklearn
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            classification_report
        )
    except Exception:
        print("[Day5] 尚未安裝 scikit-learn，請先安裝：python -m pip install scikit-learn")
        return ARTIFACTS

    # 模型們
    models = [
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("gnb", GaussianNB()),
    ]

    results = []
    roc_sources = {}

    # 先清空/建立分類報告檔
    with open(CLF_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Day5 Classification Reports\n")

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 機率輸出（若支援）
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
        roc_sources[name] = y_proba

        # 指標
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
        })

        # 報告（追加寫入）
        with open(CLF_REPORT_TXT, "a", encoding="utf-8") as f:
            f.write(f"\n=== {name} ===\n")
            f.write(classification_report(y_test, y_pred, digits=4))

        # 混淆矩陣圖（每個模型各存一張）
        cm_path = CM_DIR / f"confusion_matrix_{name}.png"
        _plot_confusion_matrix(y_test, y_pred, cm_path)

        # 係數表（僅對 Logistic）
        if name == "logreg":
            try:
                # 係數展平成一維，索引對應特徵名稱，並依絕對值排序
                coefs = pd.Series(model.coef_.ravel(), index=X_train.columns) \
                          .sort_values(key=lambda s: s.abs(), ascending=False)
                coef_df = coefs.rename("coef").reset_index().rename(columns={"index": "feature"})
                # 也附上截距（intercept）
                coef_df.loc[len(coef_df)] = {"feature": "intercept", "coef": float(model.intercept_.ravel()[0])}
                coef_df.to_csv(COEF_CSV, index=False, encoding="utf-8-sig")
            except Exception as e:
                print(f"[Day5] 無法輸出係數（Logistic）：{e}")

    # 存 metrics & ROC
    pd.DataFrame(results).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    _plot_roc(y_test, roc_sources, ROC_PNG)

    print(f"[Day5] ✅ 完成輸出：\n- {METRICS_CSV}\n- {CLF_REPORT_TXT}\n- {ROC_PNG}\n- {CM_DIR}/*.png\n- {COEF_CSV}（若 Logistic 成功輸出）")
    return ARTIFACTS

if __name__ == "__main__":
    run_day5()
