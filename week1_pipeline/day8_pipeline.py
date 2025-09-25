import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 字型設定（可選）
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 從設定檔讀取路徑與欄位
try:
    from project_config import OUTPUT_CSV_DAY4 as INPUT_CSV_PATH
except Exception:
    INPUT_CSV_PATH = Path("data_lung/processed/day4_cleaned.csv")

try:
    from project_config import TARGET_COL as TARGET_COLUMN_NAME
except Exception:
    TARGET_COLUMN_NAME = "LUNG_CANCER"

# Day8 成品資料夾
ARTIFACTS_DIR = Path("artifacts_day8")
GRIDSEARCH_RESULTS_CSV = ARTIFACTS_DIR / "gridsearch_results.csv" # 儲存 Grid Search 或超參數調整的每組結果
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl" # 最佳模型的完整物件

# 工具函式 pipeline 的防呆檢查，確保輸入資料正確

def check_target_column(dataframe: pd.DataFrame, target_column: str) -> None:
    """確認目標欄位是否存在且至少兩類（可分類）。"""
    if target_column not in dataframe.columns:
        raise ValueError(f"[Day8] 找不到目標欄：{target_column}")
    n_unique = dataframe[target_column].nunique(dropna=True)
    if n_unique < 2:
        raise ValueError(f"[Day8] 目標欄 `{target_column}` 類別數為 {n_unique}，無法做分類。")

def prepare_features_and_target(dataframe: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """把輸入的資料表拆成特徵 X 與標籤 y,並確保特徵為數值型。"""
    categorical_columns = [c for c in dataframe.select_dtypes(include="object").columns if c != target_column]
    # 找出 DataFrame 中所有 類別型欄位（object），但排除 target_column
    if categorical_columns:
        dataframe = pd.get_dummies(
            dataframe,
            columns=categorical_columns,
            drop_first=True,
            dummy_na=True
        )
        print(f"[Day8] 已 one-hot 編碼欄位：{categorical_columns}")
        """ drop_first=True → 避免虛擬變數陷阱(Dummy Variable Trap)。
            dummy_na=True → 針對缺失值也建一個欄位，例如 Gender_NaN。 """

    target_series = dataframe[target_column]   # y標籤
    feature_dataframe = dataframe.drop(columns=[target_column]) # X 特徵矩陣
    feature_dataframe = feature_dataframe.replace([np.inf, -np.inf], np.nan).fillna(0)
    # 避免出現數值爆掉（例如 log 運算後的 inf）導致模型無法訓練

    return feature_dataframe, target_series
"""  回傳 (X, y)，可直接送進 train_test_split 或 GridSearchCV """


# 主流程：Day8 超參數搜尋 / GridSearch

def run_day8(
    input_csv_path: str | Path = INPUT_CSV_PATH,
    target_column: str = TARGET_COLUMN_NAME,
    test_size: float = 0.2,
    random_state: int = 42,
):
  
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv_path)
    print(f"[Day8] 讀入：{input_csv_path} shape={df.shape}")

    check_target_column(df, target_column)

    if df[target_column].dtype == object:
        mapper = {"YES":1,"NO":0,"Y":1,"N":0,"TRUE":1,"FALSE":0,"是":1,"否":0,"1":1,"0":0,"1.0":1,"0.0":0}
        df[target_column] = df[target_column].astype(str).str.strip().str.upper().map(mapper)

    X, y = prepare_features_and_target(df, target_column)# 把資料切成特徵 X 與標籤 y

    try:
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.ensemble import RandomForestClassifier
        import joblib
    except Exception:
        print("[Day8] 尚未安裝 scikit-learn 或 joblib,請先安裝。")
        return ARTIFACTS_DIR
    """ 載入:GridSearchCV(做參數搜尋)、train_test_split(切資料）、
    RandomForestClassifier(示範用模型)、joblib(把最佳模型存成檔案）。 """

    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # 範例：隨機森林超參數搜尋
    param_grid = {
        "n_estimators": [100, 200, 300], # 隨機森林裡的樹數量（越多越穩定，但訓練時間越久）
        "max_depth": [None, 5, 10], # 樹的最大深度
        "min_samples_split": [2, 5, 10], # 節點切分所需的最少樣本數。數字越大 → 模型越保守、不容易分太細。
    }
    model = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    """ random_state=random_state:確保結果可重現。
    class_weight="balanced"：自動調整權重，應對資料集不平衡（例如陽性/陰性樣本比例差很多） """
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=2)
    """ GridSearchCV(網格搜尋交叉驗證）：
            model:要調的模型（這裡是隨機森林）。
            param_grid:上面定義的超參數組合。
            cv=5:做 5 折交叉驗證，每組參數跑 5 次，取平均。
            scoring="f1"：以 F1-score 當評估標準（平衡 precision 與 recall)。
            n_jobs=-1:用電腦所有 CPU 核心並行運算，加速計算。
            verbose=2:輸出訓練過程進度。 """


    print("[Day8] 開始 GridSearchCV...")
    grid_search.fit(X_train, y_train)
    """ fit(X_train, y_train)：會自動跑完所有參數組合 + 交叉驗證，最後存下最佳參數與最佳模型。 """

    """ 1. 定義要調整的超參數 (param_grid)。
        2. 用 GridSearchCV 系統性嘗試所有組合。
        3. 每組參數會跑 5 折交叉驗證，計算 F1-score。
        4. 找出最佳參數組合(grid_search.best_params_)。
        5. 保存最佳模型（後面會用 joblib.dump() 存成 .pkl)。 """



    # 儲存結果
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(GRIDSEARCH_RESULTS_CSV, index=False, encoding="utf-8-sig")

    # 儲存最佳模型
    best_model = grid_search.best_estimator_ # 自動抓出 在交叉驗證裡表現最好的模型（最佳參數已經套用進去）
    joblib.dump(best_model, BEST_MODEL_PATH) # 把最佳模型存成檔案：

    print(f"[Day8] ✅ 完成。最佳模型：{grid_search.best_params_}")
    print(f"- {GRIDSEARCH_RESULTS_CSV}")
    print(f"- {BEST_MODEL_PATH}")

    return ARTIFACTS_DIR

if __name__ == "__main__":
    run_day8()
