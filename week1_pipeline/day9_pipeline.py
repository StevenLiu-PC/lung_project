# encoding: utf-8
# eol: LF
# summary: feat(day9): 最終測試、ROC/CM 圖與報告

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 專案根目錄設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 匯入 Day4 輸出的清理資料
try:
    from project_config import OUTPUT_CSV_DAY4 as INPUT_CSV_PATH
except Exception:
    INPUT_CSV_PATH = Path("data_lung/processed/day4_cleaned.csv")

try:
    from project_config import TARGET_COL as TARGET_COLUMN_NAME
except Exception:
    TARGET_COLUMN_NAME = "LUNG_CANCER"

# Day9 成果輸出路徑
ARTIFACTS_DIR = Path("artifacts_day9")
REPORT_TXT = ARTIFACTS_DIR / "final_model_report.txt"
ROC_PNG = ARTIFACTS_DIR / "final_model_roc.png"
CM_PNG = ARTIFACTS_DIR / "final_model_confusion_matrix.png"

BEST_MODEL_PATH = Path("artifacts_day8/best_model.pkl")

# 工具函式

def check_target_column(dataframe: pd.DataFrame, target_column: str) -> None:
    """確認目標欄位是否存在且至少兩類。"""
    if target_column not in dataframe.columns:
        raise ValueError(f"[Day9] 找不到目標欄：{target_column}")
    if dataframe[target_column].nunique(dropna=True) < 2:
        raise ValueError(f"[Day9] 目標欄 `{target_column}` 類別不足，無法分類。")

def prepare_features_and_target(dataframe: pd.DataFrame, target_column: str):
    """拆分特徵與標籤，並處理 one-hot / 缺值。"""
    categorical_columns = [c for c in dataframe.select_dtypes(include="object").columns if c != target_column]
    if categorical_columns:
        dataframe = pd.get_dummies(
            dataframe,
            columns=categorical_columns,
            drop_first=True,
            dummy_na=True
        )
        print(f"[Day9] One-hot 編碼欄位：{categorical_columns}")

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y

def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    """繪製混淆矩陣。"""
    from sklearn.metrics import confusion_matrix # 只在函式內部才需要，用局部匯入
    import itertools # 幫忙在格子中逐一標註數字
    cm = confusion_matrix(y_true, y_pred) # 計算混淆矩陣（預設未正規化，整數計數）

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues) # 用影像方式顯示矩陣
    plt.title("Confusion Matrix (Day9)")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # 在每個格子中間印出數值
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 確保輸出資料夾存在
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_roc_curve(y_true, y_proba, out_path: Path):
    """ roc_curve() 需要傳入真實標籤 (y_true) 與預測機率 (y_proba) """
    from sklearn.metrics import roc_curve, auc # 匯入計算 ROC 曲線和 AUC 的工具
    fpr, tpr, _ = roc_curve(y_true, y_proba) 
    # auc() 計算 ROC 曲線下面積 (Area Under Curve)，衡量模型辨別能力
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.3f})") # 畫出 ROC 曲線
    plt.plot([0, 1], [0, 1], "--", color="gray")  # 畫出對角線 (隨機猜測的基準線)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Day9)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    """ ROC 曲線：看模型在不同「判斷閾值」下，真陽性率 (TPR) 與假陽性率 (FPR) 的平衡。
        AUC 值:ROC 曲線下面積，數值介於 0.5(亂猜）到 1.0(完美分類）。越接近 1 越好。
        對角線:表示隨機猜測的基準線(AUC=0.5)。 """
    

# 主流程

def run_day9(
    input_csv_path: str | Path = INPUT_CSV_PATH,
    target_column: str = TARGET_COLUMN_NAME,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Day9: 使用 Day8 找到的最佳模型，進行最終測試與輸出報告。"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    #確保 TIFACTS_DIR 這個資料夾存在，如果不存在就一路建起來；如果已經存在，就忽略，不要報錯。

    # 讀資料
    df = pd.read_csv(input_csv_path)
    print(f"[Day9] 讀入：{input_csv_path}, shape={df.shape}")# - DataFrame 的形狀 (df.shape)
    check_target_column(df, target_column)

    # 簡單轉換 YES/NO → 1/0
    if df[target_column].dtype == object: #   # 如果目標欄位是文字型別（object），先做字串轉換與標準化
        mapper = {"YES": 1, "NO": 0, "Y": 1, "N": 0, "TRUE": 1, "FALSE": 0, "是": 1, "否": 0}
        df[target_column] = df[target_column].astype(str).str.strip().str.upper().map(mapper)

    # 拆分特徵 X 與標籤 y（會處理 one-hot 與缺值）
    X, y = prepare_features_and_target(df, target_column)

    # train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    ) # 把資料集拆成 訓練集 (X_train, y_train) 和 測試集 (X_test, y_test)

    # 載入最佳模型
    import joblib
    if not BEST_MODEL_PATH.exists():
        print("[Day9] 找不到 Day8 最佳模型，請先執行 Day8。")
        return ARTIFACTS_DIR # 如果沒有模型，就直接結束函式，回傳輸出資料夾路徑
    best_model = joblib.load(BEST_MODEL_PATH) # 載入最佳模型（Day8 存下來的模型檔）

    # 預測
    y_pred = best_model.predict(X_test) # 直接產生預測標籤（0/1）
    try:
        y_proba = best_model.predict_proba(X_test)[:, 1] # [:, 1] 代表取「屬於正類別」的機率
    except Exception:
        y_proba = None
        # 如果模型不支援 predict_proba（例如 SVM 沒開啟 probability）
        # 就把 y_proba 設成 None，避免程式報錯

    # 指標
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
    acc = accuracy_score(y_test, y_pred) # 整體答對比例。
    prec = precision_score(y_test, y_pred, zero_division=0) # 真的有多少是對的
    rec = recall_score(y_test, y_pred, zero_division=0) # 真的我抓到多少
    f1 = f1_score(y_test, y_pred, zero_division=0) # Precision 與 Recall 的折衷，避免只看一邊
    auc_score = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan 
                                            #模型整體區分能力，0.5=亂猜，1=完美分類。

    # 輸出報告
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Day9 Final Model Report\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
        f.write(f"ROC-AUC  : {auc_score:.4f}\n\n")
        f.write("=== Classification Report ===\n")
        f.write(classification_report(y_test, y_pred, digits=4))
              # 內容包含 precision/recall/f1/支援度 (support)，四位小數格式

    # 混淆矩陣 & ROC
    plot_confusion_matrix(y_test, y_pred, CM_PNG)
    if y_proba is not None:
        plot_roc_curve(y_test, y_proba, ROC_PNG)

    print(f"[Day9] ✅ 完成最終模型測試，輸出：\n- {REPORT_TXT}\n- {CM_PNG}\n- {ROC_PNG}")
    return ARTIFACTS_DIR


if __name__ == "__main__":
    run_day9()
