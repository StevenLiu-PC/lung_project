import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 字型設定
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 路徑設定（優先吃 project_config，否則用預設）
try:
    from project_config import OUTPUT_CSV_DAY4 as INPUT_CSV_PATH  # Day4 的輸出
except Exception:
    INPUT_CSV_PATH = Path("data_lung/processed/day4_cleaned.csv")

try:
    from project_config import TARGET_COL as TARGET_COLUMN_NAME
except Exception:
    TARGET_COLUMN_NAME = "LUNG_CANCER"
""" 彈性化：
        如果有 project_config.py,程式就能共用同一份設定（不怕硬編路徑）。
        如果沒有 project_config.py,也能靠 except 提供安全的預設值，程式不會掛掉。 """

# Day7 輸出成品資料夾與檔名
ARTIFACTS_DIR = Path("artifacts_day7") # 定義一個主資料夾 artifacts_day7，存放 Day7 產生的所有成果
METRICS_CSV = ARTIFACTS_DIR / "metrics.csv" # 存放「各模型的整體數值表現」
CLASSIFICATION_REPORT_TXT = ARTIFACTS_DIR / "classification_report.txt" # 存放「每個模型的詳細分類報告」
ROC_PNG = ARTIFACTS_DIR / "roc_curve.png" # 存放「ROC 曲線圖」
CM_DIR = ARTIFACTS_DIR / "confusion_matrices" # 一個資料夾，存放各模型的「混淆矩陣圖」(confusion matrix)
COEF_CSV_DIR = ARTIFACTS_DIR / "coefficients" # 存放「係數表」，通常是線性模型 (例如 Logistic Regression, Linear SVM) 的權重
FI_CSV_DIR = ARTIFACTS_DIR / "feature_importances" # 存放「特徵重要性表」

# 工具：檢查目標欄
def check_target_column(dataframe: pd.DataFrame, target_column: str) -> None:
    """確認目標欄位是否存在，且至少兩類（可分類）。"""
    if target_column not in dataframe.columns:
        raise ValueError(f"[Day7] 找不到目標欄：{target_column}")
    # 如果目標欄位不在資料表的欄位清單裡，直接丟出錯誤（ValueError）
    n_unique = dataframe[target_column].nunique(dropna=True)
    # 計算這個欄位裡「有幾種不同的值」
    if n_unique < 2:
        raise ValueError(f"[Day7] 目標欄 `{target_column}` 類別數為 {n_unique}，無法做分類。")
    # 如果唯一值少於 2，代表資料只有一種標籤，無法訓練分類模型

# 工具：將類別欄位 one-hot（排除目標欄），處理 ±inf/NaN，回傳 (X, y)
def prepare_features_and_target(dataframe: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """把輸入的資料表拆成特徵 X 與標籤 y,並確保特徵為數值型(one-hot + 缺值處理）。"""
    categorical_columns = [c for c in dataframe.select_dtypes(include="object").columns if c != target_column]
            # 找出所有「字串型欄位」（類別特徵），但排除目標欄。
    if categorical_columns:
        dataframe = pd.get_dummies(
            dataframe,
            columns=categorical_columns,
            drop_first=True,   # 避免虛擬變數陷阱（Dummy Variable Trap）
            dummy_na=True      # 類別缺失值也成為一個欄位（例如 Gender_NaN）
        )
        print(f"[Day7] 已 one-hot 編碼欄位：{categorical_columns}")
        # 如果有類別型欄位，先用 one-hot 編碼 轉換成數值：

    target_series = dataframe[target_column]         # y：目標欄位（例如 LUNG_CANCER 0/1）
    feature_dataframe = dataframe.drop(columns=[target_column])  # X：其餘所有特徵

    # 將 ±inf 轉為 NaN，再以 0 補缺（保守作法，避免崩潰）
    feature_dataframe = feature_dataframe.replace([np.inf, -np.inf], np.nan).fillna(0)
    """ 先把無限大 np.inf / 無限小 -np.inf 轉成缺值(NaN)。
        然後把 NaN 全部補成 0,避免模型在 .fit() 時崩潰。 """
    return feature_dataframe, target_series
    # 輸出 (X, y)：乾淨的特徵表、標籤序列。

# 工具：簡易 train/test split（與 Day5 一致）

def train_test_split_simple(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    # test_size=0.2：測試集比例（20%）
    # random_state=42：固定亂數種子，確保每次分割結果相同
    """手工切分訓練/測試集（固定亂數種子保可重現）。"""
    n = len(X) # 算總樣本數
    indices = np.arange(n) # 建立一個索引陣列 [0, 1, 2, ..., n-1]，代表每一筆資料的位置
    rng = np.random.default_rng(seed=random_state)
    # 建立亂數產生器 (rng = random number generator)，塞入 random_state 來固定隨機性。
    rng.shuffle(indices) # 把索引陣列隨機打亂。
    cut = int((1 - test_size) * n) #算切分點（有多少資料當訓練集）。
    train_idx, test_idx = indices[:cut], indices[cut:] # 前 80% 當訓練集索引，後 20% 當測試集索引。
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

# 工具：畫多模型 ROC
def plot_roc_multi(y_test, y_proba_dict: dict[str, np.ndarray], out_path: Path):
    plt.figure()    # 建立一個新的圖 (plt.figure())
    try:
        from sklearn.metrics import roc_curve, auc as _auc # 匯入 roc_curve 和 auc（AUC 計算），暫時命名 _auc
        plotted = False # 用來標記是否真的畫出了 ROC
        for name, prob in y_proba_dict.items(): 
            # y_proba_dict：一個字典，key 是模型名稱，value 是預測的機率（predict_proba 的結果）。
            if prob is None: # 跳過 None（有些模型沒有機率輸出）
                continue
            fpr, tpr, _ = roc_curve(y_test, prob) # 算出這個模型的 ROC 曲線座標 (FPR/TPR)
            plt.plot(fpr, tpr, label=f"{name} (AUC={_auc(fpr, tpr):.3f})")
            # 畫 ROC 曲線，標籤加上模型名稱和 AUC 分數
            plotted = True
        if plotted:
            plt.plot([0, 1], [0, 1], "--") # 如果有曲線畫出來，額外加一條對角線 (y=x)，代表隨機分類
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (Day7 Models)")
            plt.legend()
            plt.tight_layout() # 自動調整版面避免文字擠壓。
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150)
        else:
            print("[Day7] 無可用機率輸出，略過 ROC 繪圖。")
    finally:
        plt.close() # 無論成功或失敗，都會把圖表關閉，避免記憶體爆掉

# 工具：畫混淆矩陣
def plot_confusion_matrix(y_test, y_pred, out_path: Path): # y_test：測試集的真實標籤。y_pred：模型預測的標籤。
    from sklearn.metrics import confusion_matrix
    import itertools  # 二分類情況下會是一個 2×2 的矩陣
    cm = confusion_matrix(y_test, y_pred) # itertools.product 用來之後迴圈遍歷矩陣的座標 (i, j)
    plt.figure()
    plt.imshow(cm, interpolation="nearest") # 確保每個格子清晰顯示，不模糊
    plt.title("Confusion Matrix")
    plt.colorbar() # 旁邊加顏色條，數字大小會有顏色強弱
    tick_marks = np.arange(cm.shape[0]) # tick_marks：生成標籤座標，例如 [0,1]
    plt.xticks(tick_marks, tick_marks); plt.yticks(tick_marks, tick_marks)
    # 設定 X/Y 軸刻度為 類別標籤（這裡假設標籤是 0,1）
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center")
        """ 雙迴圈，把每個格子 (i,j) 的數值 (cm[i,j]) 印在圖片中央。
            ha="center", va="center" → 水平、垂直都置中 """
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

# 工具：輸出線性模型係數
def export_linear_coefficients(model, feature_names: list[str], out_dir: Path, model_name: str):
    """ model:已訓練好的線性模型（必須有 coef_ 和 intercept_ 屬性）。
    feature_names:特徵名稱清單，對應到 coef_ 的順序。 """
    try:
        coefs = pd.Series(model.coef_.ravel(), index=feature_names).sort_values(key=lambda s: s.abs(), ascending=False)
        # model.coef_：線性模型的係數（每個特徵一個）。.ravel()：把係數展平成一維陣列
        # .sort_values(key=lambda s: s.abs(), ascending=False)：按照絕對值大小排序 → 最重要的特徵排最前面
        coef_df = coefs.rename("coefficient").reset_index().rename(columns={"index": "feature"})
        # coefficient：對應的係數
        coef_df.loc[len(coef_df)] = {"feature": "intercept", "coefficient": float(model.intercept_.ravel()[0])}
        # model.intercept_：線性模型的截距項。.ravel()[0]：展平後取第一個值（通常只有一個）。
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}_coefficients.csv"
        coef_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[Day7] 無法輸出線性係數（{model_name}):{e}")
        """ 若模型不支援 coef_ 或 intercept_(例如樹模型)，就會跳錯誤。
        程式不會中斷，只會印出警告。 """

# 輸出樹模型特徵
def export_tree_feature_importance(model, feature_names: list[str], out_dir: Path, model_name: str):
    # 定義一個函式 export_tree_feature_importance
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return # 如果讀不到Logistic Regression沒這個屬性，就直接 return 跳過
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        # 建立 pandas.Series：把數值型的「重要度」配上對應的「特徵名稱
        # sort_values(ascending=False)：由大到小排序，讓最重要的特徵排在上面
        out_dir.mkdir(parents=True, exist_ok=True) 
        # 確保輸出資料夾存在（mkdir(..., exist_ok=True)
        out_path = out_dir / f"{model_name}_feature_importance.csv"
        fi.rename("importance").reset_index().rename(columns={"index": "feature"}).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
    except Exception as e:
        print(f"[Day7] 無法輸出特徵重要度（{model_name}):{e}")
    """ 篩選特徵：看到哪些特徵貢獻最大，可以決定保留或刪除。
        解釋模型：幫助非技術人員理解「模型主要是根據哪些變數來判斷」。
        可視化分析：後續還能拿這份 CSV 畫長條圖(feature importance ranking) """

# 主流程
def run_day7(
    input_csv_path: str | Path = INPUT_CSV_PATH,
    target_column: str = TARGET_COLUMN_NAME,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """執行 Day 7:擴充多個模型，統一訓練、評估與產出圖表/報告/指標。"""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 讀資料
    df = pd.read_csv(input_csv_path)
    print(f"[Day7] 讀入：{input_csv_path} shape={df.shape}")

    # 目標欄檢查與簡單容錯（若 Day4 已處理，這裡通常 OK）
    check_target_column(df, target_column)
    if df[target_column].dtype == object:
        mapper = {"YES":1,"NO":0,"Y":1,"N":0,"TRUE":1,"FALSE":0,"是":1,"否":0,"1":1,"0":0,"1.0":1,"0.0":0}
        df[target_column] = df[target_column].astype(str).str.strip().str.upper().map(mapper)

    X, y = prepare_features_and_target(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split_simple(X, y, test_size, random_state)
    
    """  X_train, y_train 用來訓練模型；
         X_test, y_test 保留做最終評估（泛化能力）。 """

    # 匯入 sklearn 與可選外掛
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
        )
    except Exception:
        print("[Day7] 尚未安裝 scikit-learn,請先安裝:python -m pip install scikit-learn")
        return ARTIFACTS_DIR
    """ 模型池:Day7 想同時比較多種演算法(Logistic、Naive Bayes、樹模型、SVM、KNN 等）。
        指標池：匯入一整組分類評估指標，確保每個模型都能用相同標準來比較。
        容錯：如果 scikit-learn 沒安裝，就直接提示使用者要安裝，不讓程式崩潰。 """

    # 可選：XGBoost / LightGBM
    xgb = None
    lgbm = None
    # 先把變數 xgb 和 lgbm 設成 None，當作「預設不存在」
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=random_state
        )
    except Exception:
        pass
    """ 嘗試匯入 XGBoost。
                n_estimators=300 → 樹的數量（迭代次數）
                learning_rate=0.05 → 每次更新的步伐（小步慢走，比較穩定）
                max_depth=4 → 單棵樹的最大深度（避免過度擬合）
                subsample=0.9 → 每次隨機取 90% 的樣本訓練
                colsample_bytree=0.9 → 每次隨機取 90% 的特徵
                eval_metric="logloss" → 評估標準設為對數損失（分類常用）
                random_state=random_state → 固定隨機種子，確保結果可重現 """
    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            random_state=random_state
        )
    except Exception:
        pass
    """ 嘗試匯入 XGBoost。
                n_estimators=300 → 樹的數量（迭代次數）
                learning_rate=0.05 → 每次更新的步伐（小步慢走，比較穩定）
                max_depth=4 → 單棵樹的最大深度（避免過度擬合）
                subsample=0.9 → 每次隨機取 90% 的樣本訓練
                colsample_bytree=0.9 → 每次隨機取 90% 的特徵
                eval_metric="logloss" → 評估標準設為對數損失（分類常用）
                random_state=random_state → 固定隨機種子，確保結果可重現 """

    # 模型清單
    models: list[tuple[str, object]] = [ # 可以用迴圈自動訓練、比較多個模型
        ("logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced")), # 自動調整類別權重，處理不平衡樣本
        ("gaussian_naive_bayes", GaussianNB()), # 朴素貝葉斯，假設特徵符合常態分布
        ("decision_tree", DecisionTreeClassifier(random_state=random_state, class_weight="balanced")), # 可輸出 特徵重要性 (feature_importances_)
        ("random_forest", RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")), 
        # 多棵隨機決策樹投票，比單棵樹穩健，越多越穩，但訓練較慢
        ("gradient_boosting", GradientBoostingClassifier(random_state=random_state)), # 一種提升法（Boosting），逐步修正前一棵樹的錯誤
        ("svc_rbf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
        # 核心在於 RBF kernel,可以處理非線性資料。probability=True:讓模型能輸出機率需要計算 ROC。缺點:大數據集會很慢。
        ("knn", KNeighborsClassifier(n_neighbors=7)),
    ]
    if xgb is not None:
        models.append(("xgboost", xgb))
    if lgbm is not None:
        models.append(("lightgbm", lgbm))
        """  自動迭代訓練所有模型。
             統一收集指標accuracy、precision、recall、f1、ROC AUC。
             額外輸出係數Logistic、特徵重要性Tree/XGB/LGBM。 """

    # 先清空/建立分類報告檔
    with open(CLASSIFICATION_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Day7 Classification Reports\n")

    results = [] # 一個空清單，用來存放每個模型的評估指標（accuracy、precision、recall、f1、roc_auc）
    roc_sources = {} # 一個字典，存每個模型的 ROC 曲線用的機率輸出 (y_proba)，之後畫多模型 ROC 曲線要用

    for model_name, model_obj in models:
        print(f"[Day7] 訓練模型：{model_name}")
        model_obj.fit(X_train, y_train) # 把模型用訓練資料 擬合（學習參數）
        y_pred = model_obj.predict(X_test) # 用測試集做預測，得到 0/1 類別結果。
        """ 計算 accuracy / precision / recall / f1 / roc_auc
            輸出 classification_report
            畫混淆矩陣、ROC 曲線
            匯出係數 / 特徵重要性 """

        # 機率輸出（若支援）
        try:
            y_proba = model_obj.predict_proba(X_test)[:, 1] # 取出「每筆樣本屬於正類別 (label=1) 的機率」
        except Exception:
            y_proba = None
        roc_sources[model_name] = y_proba # 統一存入 roc_sources，供 ROC 曲線比較使用

        # 指標
        accuracy  = accuracy_score(y_test, y_pred) # 計算方式 = (預測正確的樣本數) ÷ (全部樣本數)
        precision = precision_score(y_test, y_pred, zero_division=0) # 計算方式 = 真陽性 ÷ (真陽性 + 假陽性)
        recall    = recall_score(y_test, y_pred, zero_division=0) # 計算方式 = 真陽性 ÷ (真陽性 + 假陰性)。
        f1        = f1_score(y_test, y_pred, zero_division=0) # 計算方式 = 2 × (Precision × Recall) ÷ (Precision + Recall)
        roc_auc   = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        """ roc_auc_score 需要「機率輸出」(y_proba)，而不是單純 0/1 預測。
                值域:0~1
                0.5 ≈ 隨機猜測
                0.7 算可接受
                0.8 算良好
                0.9 算優秀
 """
        results.append({ # 把所有評估指標打包成一個字典，加到 results list。
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }) # 轉成 DataFrame，輸出成 metrics.csv

        # 追加分類報告
        with open(CLASSIFICATION_REPORT_TXT, "a", encoding="utf-8") as f:
            f.write(f"\n=== {model_name} ===\n")
            f.write(classification_report(y_test, y_pred, digits=4))

        # 混淆矩陣 混淆矩陣能顯示「TP, FP, TN, FN」分布，比單純 accuracy 更直觀
        cm_path = CM_DIR / f"confusion_matrix_{model_name}.png"
        plot_confusion_matrix(y_test, y_pred, cm_path)

        # 線性模型係數: 解釋「哪個特徵對預測影響最大（正/負影響）
        if model_name in {"logistic_regression"}:
            export_linear_coefficients(model_obj, X_train.columns.tolist(), COEF_CSV_DIR, model_name)
        """ 輸出：每個特徵的 迴歸係數 (coefficient) + 截距 (intercept) """

        # 樹模型特徵重要度: 幫助我們知道「模型是依靠哪些特徵在做判斷
        if model_name in {"decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm"}:
            export_tree_feature_importance(model_obj, X_train.columns.tolist(), FI_CSV_DIR, model_name)

        """  畫混淆矩陣 → 視覺化 TP / FP / TN / FN。
            輸出 Logistic Regression 的係數 → 看特徵影響方向。
            輸出 Tree-based 模型的特徵重要度 → 看哪些特徵最關鍵。 """

    # 存 metrics & ROC（多模型）
    pd.DataFrame(results).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    plot_roc_multi(y_test, roc_sources, ROC_PNG)
    """ metrics.csv → 表格，方便 Excel / Pandas 分析多模型的成績。
        roc_curve.png → 視覺化比較不同模型 ROC 曲線。 """

    print(f"[Day7] ✅ 完成輸出：\n- {METRICS_CSV}\n- {CLASSIFICATION_REPORT_TXT}\n- {ROC_PNG}\n- {CM_DIR}/*.png\n- {COEF_CSV_DIR}（若線性模型）\n- {FI_CSV_DIR}（若樹模型）")
    return ARTIFACTS_DIR


if __name__ == "__main__":
    run_day7()
