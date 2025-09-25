# Day8 pipeline: 資料品質檢查 (QC) 與清理
# 註解：僅新增說明，不影響程式邏輯

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 摮?閮剖?嚗?賂?
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 敺身摰?霈?楝敺?甈?
try:
    from project_config import OUTPUT_CSV_DAY4 as INPUT_CSV_PATH
except Exception:
    INPUT_CSV_PATH = Path("data_lung/processed/day4_cleaned.csv")

try:
    from project_config import TARGET_COL as TARGET_COLUMN_NAME
except Exception:
    TARGET_COLUMN_NAME = "LUNG_CANCER"

# Day8 ??鞈?憭?
ARTIFACTS_DIR = Path("artifacts_day8")
GRIDSEARCH_RESULTS_CSV = ARTIFACTS_DIR / "gridsearch_results.csv" # ?脣? Grid Search ???隤踵??蝯???
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl" # ?雿單芋??摰?拐辣

# 撌亙?賢? pipeline ??炎?伐?蝣箔?頛詨鞈?甇?Ⅱ

def check_target_column(dataframe: pd.DataFrame, target_column: str) -> None:
    """蝣箄??格?甈??臬摮銝撠憿??臬?憿???""
    if target_column not in dataframe.columns:
        raise ValueError(f"[Day8] ?曆??啁璅?嚗target_column}")
    n_unique = dataframe[target_column].nunique(dropna=True)
    if n_unique < 2:
        raise ValueError(f"[Day8] ?格?甈?`{target_column}` 憿?貊 {n_unique}嚗瘜?????)

def prepare_features_and_target(dataframe: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """?撓?亦?鞈?銵冽??敺?X ??蝐?y,銝衣Ⅱ靽敺萇?詨澆???""
    categorical_columns = [c for c in dataframe.select_dtypes(include="object").columns if c != target_column]
    # ?曉 DataFrame 銝剜???憿??雿?object嚗?雿???target_column
    if categorical_columns:
        dataframe = pd.get_dummies(
            dataframe,
            columns=categorical_columns,
            drop_first=True,
            dummy_na=True
        )
        print(f"[Day8] 撌?one-hot 蝺函Ⅳ甈?嚗categorical_columns}")
        """ drop_first=True ???踹??霈?琿(Dummy Variable Trap)??
            dummy_na=True ????蝻箏仃?潔?撱箔???雿?靘? Gender_NaN??"""

    target_series = dataframe[target_column]   # y璅惜
    feature_dataframe = dataframe.drop(columns=[target_column]) # X ?孵噩?拚
    feature_dataframe = feature_dataframe.replace([np.inf, -np.inf], np.nan).fillna(0)
    # ?踹??箇?詨潛???靘? log ??敺? inf嚗??湔芋?瘜?蝺?

    return feature_dataframe, target_series
"""  ? (X, y)嚗?湔??train_test_split ??GridSearchCV """


# 銝餅?蝔?Day8 頞??豢?撠?/ GridSearch

def run_day8(
    input_csv_path: str | Path = INPUT_CSV_PATH,
    target_column: str = TARGET_COLUMN_NAME,
    test_size: float = 0.2,
    random_state: int = 42,
):
  
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv_path)
    print(f"[Day8] 霈?伐?{input_csv_path} shape={df.shape}")

    check_target_column(df, target_column)

    if df[target_column].dtype == object:
        mapper = {"YES":1,"NO":0,"Y":1,"N":0,"TRUE":1,"FALSE":0,"??:1,"??:0,"1":1,"0":0,"1.0":1,"0.0":0}
        df[target_column] = df[target_column].astype(str).str.strip().str.upper().map(mapper)

    X, y = prepare_features_and_target(df, target_column)# ?????敺?X ??蝐?y

    try:
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.ensemble import RandomForestClassifier
        import joblib
    except Exception:
        print("[Day8] 撠摰? scikit-learn ??joblib,隢?摰???)
        return ARTIFACTS_DIR
    """ 頛:GridSearchCV(???豢?撠??rain_test_split(??????
    RandomForestClassifier(蝷箇??冽芋???oblib(??雿單芋????獢???"""

    # ?鞈?
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # 蝭?嚗璈ㄝ?????
    param_grid = {
        "n_estimators": [100, 200, 300], # ?冽?璉格?鋆∠?璅寞??頞?頞帘摰?雿?蝺湔???銋?
        "max_depth": [None, 5, 10], # 璅寧??憭扳楛摨?
        "min_samples_split": [2, 5, 10], # 蝭暺??????撠見?祆?摮?憭???璅∪?頞?摰?摰寞??云蝝啜?
    }
    model = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    """ random_state=random_state:蝣箔?蝯??舫??整?
    class_weight="balanced"嚗?矽?湔?????鞈???撟唾﹛嚗?憒???唳扳見?祆?靘榆敺?嚗?"""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=2)
    """ GridSearchCV(蝬脫??鈭文?撽?嚗?
            model:閬矽?芋???ㄐ?舫璈ㄝ????
            param_grid:銝摰儔???蝯???
            cv=5:??5 ?漱??霅?瘥??頝?5 甈∴??像??
            scoring="f1"嚗誑 F1-score ?嗉?隡唳?皞?撟唾﹛ precision ??recall)??
            n_jobs=-1:?券?行???CPU ?詨?銝西???嚗???蝞?
            verbose=2:頛詨閮毀???脣漲??"""


    print("[Day8] ?? GridSearchCV...")
    grid_search.fit(X_train, y_train)
    """ fit(X_train, y_train)嚗??芸?頝?????貊???+ 鈭文?撽?嚗?敺?銝?雿喳??貉??雿單芋??"""

    """ 1. 摰儔閬矽?渡?頞???(param_grid)??
        2. ??GridSearchCV 蝟餌絞?批?閰行?????
        3. 瘥???? 5 ?漱??霅?閮? F1-score??
        4. ?曉?雿喳??貊???grid_search.best_params_)??
        5. 靽??雿單芋??敺? joblib.dump() 摮? .pkl)??"""



    # ?脣?蝯?
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(GRIDSEARCH_RESULTS_CSV, index=False, encoding="utf-8-sig")

    # ?脣??雿單芋??
    best_model = grid_search.best_estimator_ # ?芸?? ?其漱??霅ㄐ銵函?憟賜?璅∪?嚗?雿喳??詨歇蝬??券脣嚗?
    joblib.dump(best_model, BEST_MODEL_PATH) # ??雿單芋????獢?

    print(f"[Day8] ??摰???雿單芋??{grid_search.best_params_}")
    print(f"- {GRIDSEARCH_RESULTS_CSV}")
    print(f"- {BEST_MODEL_PATH}")

    return ARTIFACTS_DIR

if __name__ == "__main__":
    run_day8()

