# encoding: utf-8
# eol: LF
# summary: feat(day6): 交叉驗證骨架與報告

import sys
from pathlib import Path

# week1_pipeline/xxx.py → 專案根 (lung_project)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 字型設定（可視需要保留）
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 從設定檔讀取 Day4 路徑與目標欄；若失敗用預設
try:
    from project_config import OUTPUT_CSV_DAY4 as INPUT_CSV_PATH  # Day4 的輸出
except Exception:
    INPUT_CSV_PATH = Path("data_lung/processed/day4_cleaned.csv")

try:
    from project_config import TARGET_COL as TARGET_COLUMN_NAME
except Exception:
    TARGET_COLUMN_NAME = "LUNG_CANCER"

# Day6 輸出成品資料夾與檔名
ARTIFACTS_DIR = Path("artifacts_day6")
CV_METRICS_CSV = ARTIFACTS_DIR / "cv_metrics.csv"
CV_REPORT_TXT = ARTIFACTS_DIR / "cv_classification_reports.txt"

# ----------------------------------------------------------------------
# 工具：檢查目標欄是否存在、且類別數足夠（至少兩類）
# ----------------------------------------------------------------------
def check_target_column(dataframe: pd.DataFrame, target_column: str) -> None:
    """確認目標欄位是否存在，並且確定它的類別數是否足夠做分類。"""
    if target_column not in dataframe.columns:
        raise ValueError(f"[Day6] 找不到目標欄：{target_column}")
    n_unique = dataframe[target_column].nunique(dropna=True)
    # 如果類別數小於 2 → 代表整欄都是同一個值（例如全是 1），無法訓練分類模型
    if n_unique < 2:
        raise ValueError(f"[Day6] 目標欄 `{target_column}` 類別數為 {n_unique}，無法做分類。")

# 工具：將類別欄位 one-hot（排除目標欄），處理 ±inf/NaN，回傳 (X, y)
def prepare_features_and_target(dataframe: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    #把原始資料表拆成特徵 X與標籤 y
    categorical_columns = [c for c in dataframe.select_dtypes(include="object").columns if c != target_column]
    #從資料表裡找出資料型態為字串（object）的欄位，這些通常代表類別型特徵（如性別、地區…）。同時排除目標欄（target_column），避免把 y 拿去做特徵編碼。
    if categorical_columns:
        dataframe = pd.get_dummies(
            dataframe,
            columns=categorical_columns,
            drop_first=True,   # 避免虛擬變數陷阱（Dummy Variable Trap）
            dummy_na=True      # 類別缺失值也成為一個欄位（例如 Gender_NaN）
        )
        print(f"[Day6] 已 one-hot 編碼欄位：{categorical_columns}")
        #僅做紀錄與除錯：在終端顯示哪些欄位被做了 one-hot，方便回頭檢查流程。
        #用 one-hot 編碼把類別欄位變成 0/1 欄位模型才能吃。

    target_series = dataframe[target_column]  #從資料表中取出目標欄位，存成 target_series y：（例如 LUNG_CANCER 0/1）
    feature_dataframe = dataframe.drop(columns=[target_column])  # 把目標欄位刪掉，剩下的就是特徵 X。

    # 先把無限大/小改成缺值  → 把缺值用 0 補上，不然整個模型 .fit() 會直接崩潰。
    feature_dataframe = feature_dataframe.replace([np.inf, -np.inf], np.nan).fillna(0)

    return feature_dataframe, target_series
    #feature_dataframe → 特徵 X（已經確保沒有 NaN/±inf）    target_series → 標籤 y


# 工具：繪製「平均 ROC 曲線」給單一模型（若各折有 predict_proba）

def plot_mean_roc(all_fprs: list[np.ndarray], all_tprs: list[np.ndarray], out_path: Path, title: str) -> None:
    """
    all_fprs:每一折交叉驗證算出來的 FPR假陽性率, False Positive Rate
    all_tprs:每一折的 TPR真陽性率, True Positive Rate,也是一個 ndarray 的清單。
    說明：
      - 使用每一折的假陽性率(FPR)與真陽性率(TPR)
      - 內插到共同的 FPR 網格(0~1),再逐點取平均 → 得到平均 ROC
    """
    if not all_fprs or not all_tprs:
        print(f"[Day6] 沒有可用的 FPR/TPR,略過平均 ROC 圖：{title}")
        return
    #如果沒有 FPR 或 TPR（例如模型不支援 predict_proba）就直接跳過，不要畫圖。

    common_fpr = np.linspace(0, 1, 200)
    tpr_stack = []
    """ np.linspace(0, 1, 200) → 在 0 到 1 之間，平均切 200 個點。
        意思：我要建立一條「共同的 X 軸FPR」來對齊。
        tpr_stack → 用來存放「每一折的內插後 TPR」。
     """

    from numpy import interp
    for fpr, tpr in zip(all_fprs, all_tprs):
        interp_tpr = interp(common_fpr, fpr, tpr)
        tpr_stack.append(interp_tpr)
    
    """ zip(all_fprs, all_tprs) → 把 FPR 和 TPR 一一配對（同一折的數據）。
        interp(common_fpr, fpr, tpr) →
                把原本「稀疏不一樣的 FPR-TPR 點」拉伸到 共同的 200 個 FPR 節點。
                這樣才可以逐點平均，不會對不齊。
        append → 把每一折的內插後 TPR 存進堆疊。
    """
    mean_tpr = np.mean(np.vstack(tpr_stack), axis=0)
    """ np.vstack(tpr_stack) → 把所有折的 TPR 堆成一個矩陣（每一列是一折）。
        np.mean(..., axis=0) → 對每個 FPR 節點，算所有折的平均 TPR。
        結果就是一條「平均 ROC 曲線」。 
    """
    plt.figure()
    plt.plot(common_fpr, mean_tpr, label="Mean ROC")  #x 軸 = common_fpr（0~1 的標準格子）。y 軸 = mean_tpr（每格取平均的真陽性率）
    plt.plot([0, 1], [0, 1], "--")  # 參考對角線（隨機分類）
    plt.xlabel("False Positive Rate") # x 軸 = 假陽性率 (FPR)
    plt.ylabel("True Positive Rate")  # y 軸 = 真陽性率 (TPR)。
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    #確保輸出的資料夾存在（沒有就自動建），然後存圖到 out_path 路徑，解析度 150 dpi。
    plt.close()

# 主流程：交叉驗證
def run_day6(
    input_csv_path: str | Path = INPUT_CSV_PATH, #資料來源路徑
    target_column: str = TARGET_COLUMN_NAME,  #目標欄位名稱
    n_splits: int = 5,   #K 折交叉驗證的「折數」
    random_state: int = 42, #隨機種子。固定這個值，讓折分結果可重現
) -> Path:
    """
    執行交叉驗證流程：
      - 讀 Day4 清理輸出
      - 檢查/正規化目標欄
      - one-hot 類別特徵、處理缺值
      - 使用 StratifiedKFold 做 K 折交叉驗證，計算各折指標
      - 若模型支援 predict_proba,額外繪製「平均 ROC」
      - 輸出:cv_metrics.csv、cv_classification_reports.txt、(模型)平均 ROC 圖
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True) #如果中間的父層資料夾不存在，就一路幫你建起來

    # 讀資料
    df = pd.read_csv(input_csv_path)
    print(f"[Day6] 讀入：{input_csv_path} shape={df.shape}")

    # 目標欄檢查與簡單容錯（若 Day4 已處理，這裡通常 OK）
    check_target_column(df, target_column) #呼叫前面定義好的工具函式
    if df[target_column].dtype == object: #目標欄位存在 ,目標欄位至少有 2 種不同的值
        # 嘗試將 YES/NO 轉為 1/0（保險用）
        mapper = {"YES":1,"NO":0,"Y":1,"N":0,"TRUE":1,"FALSE":0,"是":1,"否":0,"1":1,"0":0,"1.0":1,"0.0":0}
        df[target_column] = df[target_column].astype(str).str.strip().str.upper().map(mapper)

    X, y = prepare_features_and_target(df, target_column)

    """ 呼叫函式 prepare_features_and_target,把資料切成:
        X = 特徵features,就是除了目標欄位以外的所有欄位。
        y = 標籤target,就是我們要預測的目標欄位。
    這一步同時會處理：
        one-hot 編碼（把字串欄位轉成數字）。
        inf/NaN 清理（避免後續模型崩潰）。 """

    # 匯入 sklearn 元件（若未安裝則提示）
    try:
        from sklearn.model_selection import StratifiedKFold 
        #交叉驗證器，確保每一折的正/負樣本比例接近整體分布（避免嚴重不平衡時有一折全是 0 或全是 1）
        from sklearn.linear_model import LogisticRegression #baseline 模型
        from sklearn.naive_bayes import GaussianNB   #Naive Bayes baseline 模型
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_curve, auc, classification_report
        )
    except Exception:
        print("[Day6] 尚未安裝 scikit-learn,請先安裝:python -m pip install scikit-learn")
        return ARTIFACTS_DIR

    # 建立要比較的模型（可依需求擴充）
    models = [
        ("logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        # max_iter=1000：允許迭代最多 1000 次   class_weight="balanced"：自動調整類別權重，處理類別不平衡問題
        ("gaussian_naive_bayes", GaussianNB()),
        # 高斯分佈的朴素貝葉斯分類器  適合處理數值型資料，假設特徵符合常態分佈。
    ]

    # 交叉驗證器（確保每一折 (fold) 的正負樣本比例（0/1）跟整體資料集相近，避免某折全是 0 沒有 1。）
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_rows = []  #放每一折 (fold) 的結果
    # 用來繪製平均 ROC 的容器（分別存放每個模型在每一折得到的 FPR / TPR）
    model_to_fprs = {name: [] for name, _ in models}
    model_to_tprs = {name: [] for name, _ in models}

    # 先清空/建立詳細報告檔
    with open(CV_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Day6 Cross-Validation Classification Reports\n")

    # 逐模型、逐折訓練與評估
    for model_name, model_obj in models:  #交叉驗證要針對「每一種模型」都跑一遍
        fold_index = 0 # 從0開始編號
        for train_index, valid_index in kfold.split(X, y):
            # train_index：這一折的訓練集樣本索引 valid_index：這一折的驗證集樣本索引
            # K折交叉驗證，把資料分成K份，輪流挑1份當驗證集，其餘當訓練集。
            fold_index += 1 #記錄目前是第幾折 (fold)，方便之後輸出結果或報告時標註

            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_valid = X.iloc[valid_index]
            y_valid = y.iloc[valid_index]
            # 依照 KFold 分好的索引，把原始資料拆成「訓練特徵、訓練標籤」與「驗證特徵、驗證標籤

            model_obj.fit(X_train, y_train) #對訓練集做「學習」，更新模型的參數
            y_pred = model_obj.predict(X_valid) #用訓練好的模型，對驗證集做預測
            """ 結果是 預測標籤 (0/1)，會拿來跟真實的 y_valid 做比較，算正確率、精確率等指標。 """
            # 嘗試取得機率（繪 ROC 需要）
            try:
                y_proba = model_obj.predict_proba(X_valid)[:, 1]
                # 讓模型輸出「機率預測」  這裡用陣列切片罹患肺癌 (1) 的機率
            except Exception:
                y_proba = None
            """ 嘗試取得模型對「類別 1」的預測機率,如果模型不支援,就先放棄,標記成 None """

            # 計算評估指標
            acc  = accuracy_score(y_valid, y_pred)  # (預測正確的筆數) ÷ (總筆數)
            prec = precision_score(y_valid, y_pred, zero_division=0) # 真陽性 ÷ (真陽性 + 假陽性)
            rec  = recall_score(y_valid, y_pred, zero_division=0) # 真陽性 ÷ (真陽性 + 假陰性)
            f1   = f1_score(y_valid, y_pred, zero_division=0) #F1 = 2 × (precision × recall) ÷ (precision + recall)
            """  zero_division=0  有時候分母可能變成 0(例如模型完全沒預測「正樣本) """
            # ROC-AUC（需要機率），無法取得機率就以 NaN 表示
            if y_proba is not None: # 如果模型沒有提供機率輸出，直接跳過 ROC 計算
                fpr, tpr, _ = roc_curve(y_valid, y_proba)# 根據真實標籤 (y_valid) 與預測機率 (y_proba) 計算 ROC 曲線座標
                roc_auc = auc(fpr, tpr)  # AUC 代表 ROC 曲線下的面積，值域 0~1，越接近 1 越好
                model_to_fprs[model_name].append(fpr)  #(假陽性率, x 軸)
                model_to_tprs[model_name].append(tpr) #(真陽性率, y 軸)
            else:
                roc_auc = np.nan 
            """ 如果模型沒機率輸出，就把 AUC 設成 NaN (Not a Number)，避免程式崩潰 """
            all_rows.append({
                "model": model_name,
                "fold": fold_index,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc_auc,
            })
            """ 把這一折的所有指標打包成一個字典，丟進 all_rows。
            最後會轉成 DataFrame (pd.DataFrame(all_rows))，再輸出到 cv_metrics.csv """

            # 寫入分類報告（方便審核每一折的表現）
            with open(CV_REPORT_TXT, "a", encoding="utf-8") as f:
                f.write(f"\n=== {model_name} | fold {fold_index} ===\n")
                f.write(classification_report(y_valid, y_pred, digits=4))

        # 模型層級的平均 ROC（若該模型各折都有 FPR/TPR 資料）
        if model_to_fprs[model_name] and model_to_tprs[model_name]:
            """ model_to_fprs[model_name]：存放這個模型 每一折的 FPR(False Positive Rate 假陽性率）。
                model_to_tprs[model_name]：存放這個模型 每一折的 TPR(True Positive Rate 真陽性率） """
            mean_roc_path = ARTIFACTS_DIR / f"cv_mean_roc_{model_name}.png"
            plot_mean_roc(model_to_fprs[model_name], model_to_tprs[model_name], mean_roc_path, f"CV Mean ROC - {model_name}")
            """ model_to_fprs[model_name] → 該模型在 K 折中每一折的 FPR。
                model_to_tprs[model_name] → 該模型在 K 折中每一折的 TPR。
                mean_roc_path → 輸出檔案路徑。
                f"CV Mean ROC - {model_name}" → 圖表標題,例如「CV Mean ROC - logistic_regression」。 """
    # 將所有折的結果彙整成表格並輸出
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(CV_METRICS_CSV, index=False, encoding="utf-8-sig")

    # 額外產出「各模型平均分數」的摘要印在終端（方便快速檢視）
    if not metrics_df.empty: # 如果完全沒有資料（例如模型沒跑成功），就不要做後續平均計算
        summary = metrics_df.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].mean().round(4)
        # 算出「交叉驗證的平均成績
        print("\n=== Day6 各模型交叉驗證平均分數 ===")
        print(summary)

    print(f"\n[Day6] ✅ 完成輸出：\n- {CV_METRICS_CSV}\n- {CV_REPORT_TXT}\n- {ARTIFACTS_DIR} 下各模型的平均 ROC 圖（若可計算）")
    return ARTIFACTS_DIR


if __name__ == "__main__":
    run_day6()
