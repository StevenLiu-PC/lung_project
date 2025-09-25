# Day7 pipeline: 多模型比較/ROC/重要度
# 註解：僅新增說明，不影響程式邏輯

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 摮?閮剖?
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 頝臬?閮剖?嚗?? project_config嚗??身嚗?
try:
    from project_config import OUTPUT_CSV_DAY4 as INPUT_CSV_PATH  # Day4 ?撓??
except Exception:
    INPUT_CSV_PATH = Path("data_lung/processed/day4_cleaned.csv")

try:
    from project_config import TARGET_COL as TARGET_COLUMN_NAME
except Exception:
    TARGET_COLUMN_NAME = "LUNG_CANCER"
""" 敶批?嚗?
        憒???project_config.py,蝔?撠梯?梁??隞質身摰?銝′蝺刻楝敺???
        憒?瘝? project_config.py,銋??except ??摰??閮剖潘?蝔?銝?????"""

# Day7 頛詨??鞈?憭曇?瑼?
ARTIFACTS_DIR = Path("artifacts_day7") # 摰儔銝?蜓鞈?憭?artifacts_day7嚗???Day7 ?Ｙ???????
METRICS_CSV = ARTIFACTS_DIR / "metrics.csv" # 摮??璅∪??擃?潸”?整?
CLASSIFICATION_REPORT_TXT = ARTIFACTS_DIR / "classification_report.txt" # 摮???芋??閰喟敦???勗???
ROC_PNG = ARTIFACTS_DIR / "roc_curve.png" # 摮?OC ?脩???
CM_DIR = ARTIFACTS_DIR / "confusion_matrices" # 銝???冗嚗??曉?璅∪??毽瘛?????confusion matrix)
COEF_CSV_DIR = ARTIFACTS_DIR / "coefficients" # 摮???貉”???虜?舐??扳芋??(靘? Logistic Regression, Linear SVM) ????
FI_CSV_DIR = ARTIFACTS_DIR / "feature_importances" # 摮?敺菟?閬扯”??

# 撌亙嚗炎?亦璅?
def check_target_column(dataframe: pd.DataFrame, target_column: str) -> None:
    """蝣箄??格?甈??臬摮嚗??喳??拚?嚗??嚗?""
    if target_column not in dataframe.columns:
        raise ValueError(f"[Day7] ?曆??啁璅?嚗target_column}")
    # 憒??格?甈?銝鞈?銵函?甈?皜鋆∴??湔銝?航炊嚗alueError嚗?
    n_unique = dataframe[target_column].nunique(dropna=True)
    # 閮???雿ㄐ??撟曄車銝??潦?
    if n_unique < 2:
        raise ValueError(f"[Day7] ?格?甈?`{target_column}` 憿?貊 {n_unique}嚗瘜?????)
    # 憒??臭??澆???2嚗誨銵刻????蝔格?蝐歹??⊥?閮毀??璅∪?

# 撌亙嚗?憿甈? one-hot嚗??斤璅?嚗??? 簣inf/NaN嚗???(X, y)
def prepare_features_and_target(dataframe: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """?撓?亦?鞈?銵冽??敺?X ??蝐?y,銝衣Ⅱ靽敺萇?詨澆?(one-hot + 蝻箏潸?????""
    categorical_columns = [c for c in dataframe.select_dtypes(include="object").columns if c != target_column]
            # ?曉???銝脣?甈???憿?孵噩嚗?雿??斤璅???
    if categorical_columns:
        dataframe = pd.get_dummies(
            dataframe,
            columns=categorical_columns,
            drop_first=True,   # ?踹??霈?琿嚗ummy Variable Trap嚗?
            dummy_na=True      # 憿蝻箏仃?潔??銝??雿?靘? Gender_NaN嚗?
        )
        print(f"[Day7] 撌?one-hot 蝺函Ⅳ甈?嚗categorical_columns}")
        # 憒????亙?甈?嚗???one-hot 蝺函Ⅳ 頧???潘?

    target_series = dataframe[target_column]         # y嚗璅?雿?靘? LUNG_CANCER 0/1嚗?
    feature_dataframe = dataframe.drop(columns=[target_column])  # X嚗擗??敺?

    # 撠?簣inf 頧 NaN嚗?隞?0 鋆撩嚗?摰?瘜??踹?撏拇蔑嚗?
    feature_dataframe = feature_dataframe.replace([np.inf, -np.inf], np.nan).fillna(0)
    """ ???⊿?憭?np.inf / ?⊿?撠?-np.inf 頧?蝻箏?NaN)??
        ?嗅???NaN ?券鋆? 0,?踹?璅∪???.fit() ?援瞏啜?"""
    return feature_dataframe, target_series
    # 頛詨 (X, y)嚗嗾瘛函??孵噩銵具?蝐文???

# 撌亙嚗陛??train/test split嚗? Day5 銝?湛?

def train_test_split_simple(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    # test_size=0.2嚗葫閰阡?瘥?嚗?0%嚗?
    # random_state=42嚗摰??貊車摮?蝣箔?瘥活?蝯??詨?
    """?極??閮毀/皜祈岫???箏?鈭蝔桀?靽?嚗?""
    n = len(X) # 蝞蜇璅???
    indices = np.arange(n) # 撱箇?銝?揣撘??[0, 1, 2, ..., n-1]嚗誨銵冽?銝蝑???雿蔭
    rng = np.random.default_rng(seed=random_state)
    # 撱箇?鈭?Ｙ???(rng = random number generator)嚗???random_state 靘摰璈扼?
    rng.shuffle(indices) # ?揣撘?璈?鈭?
    cut = int((1 - test_size) * n) #蝞???嚗?憭?鞈??嗉?蝺湧?嚗?
    train_idx, test_idx = indices[:cut], indices[cut:] # ??80% ?嗉?蝺湧?蝝Ｗ?嚗? 20% ?嗆葫閰阡?蝝Ｗ???
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

# 撌亙嚗憭芋??ROC
def plot_roc_multi(y_test, y_proba_dict: dict[str, np.ndarray], out_path: Path):
    plt.figure()    # 撱箇?銝??? (plt.figure())
    try:
        from sklearn.metrics import roc_curve, auc as _auc # ?臬 roc_curve ??auc嚗UC 閮?嚗??急??賢? _auc
        plotted = False # ?其?璅??臬???怠鈭?ROC
        for name, prob in y_proba_dict.items(): 
            # y_proba_dict嚗????賂?key ?舀芋??蝔梧?value ?舫?皜祉?璈?嚗redict_proba ??????
            if prob is None: # 頝喲? None嚗?鈭芋?????撓?綽?
                continue
            fpr, tpr, _ = roc_curve(y_test, prob) # 蝞?芋?? ROC ?脩?摨扳? (FPR/TPR)
            plt.plot(fpr, tpr, label=f"{name} (AUC={_auc(fpr, tpr):.3f})")
            # ??ROC ?脩?嚗?蝐文?銝芋??蝔勗? AUC ?
            plotted = True
        if plotted:
            plt.plot([0, 1], [0, 1], "--") # 憒??蝺?箔?嚗?憭?銝璇?閫? (y=x)嚗誨銵券璈?憿?
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (Day7 Models)")
            plt.legend()
            plt.tight_layout() # ?芸?隤踵??踹???????
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150)
        else:
            print("[Day7] ?∪?冽??撓?綽??仿? ROC 蝜芸???)
    finally:
        plt.close() # ?∟????仃???賣???銵券????踹?閮擃???

# 撌亙嚗瘛瑟??拚
def plot_confusion_matrix(y_test, y_pred, out_path: Path): # y_test嚗葫閰阡???撖行?蝐扎_pred嚗芋??皜祉?璅惜??
    from sklearn.metrics import confusion_matrix
    import itertools  # 鈭?憿?瘜??銝??2?2 ???
    cm = confusion_matrix(y_test, y_pred) # itertools.product ?其?銋?餈游??風?拚?漣璅?(i, j)
    plt.figure()
    plt.imshow(cm, interpolation="nearest") # 蝣箔?瘥摮??圈＊蝷綽?銝芋蝟?
    plt.title("Confusion Matrix")
    plt.colorbar() # ?????脫?嚗摮之撠????脣撥撘?
    tick_marks = np.arange(cm.shape[0]) # tick_marks嚗???蝐文漣璅?靘? [0,1]
    plt.xticks(tick_marks, tick_marks); plt.yticks(tick_marks, tick_marks)
    # 閮剖? X/Y 頠詨摨衣 憿璅惜嚗ㄐ?身璅惜??0,1嚗?
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center")
        """ ?艘?????摮?(i,j) ???(cm[i,j]) ?啣??銝剖亢??
            ha="center", va="center" ??瘞游像???湧蝵桐葉 """
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

# 撌亙嚗撓?箇??扳芋????
def export_linear_coefficients(model, feature_names: list[str], out_dir: Path, model_name: str):
    """ model:撌脰?蝺游末???扳芋??敹???coef_ ??intercept_ 撅祆改???
    feature_names:?孵噩?迂皜嚗?? coef_ ??摨?"""
    try:
        coefs = pd.Series(model.coef_.ravel(), index=feature_names).sort_values(key=lambda s: s.abs(), ascending=False)
        # model.coef_嚗??扳芋??靽嚗??敺萎?????ravel()嚗?靽撅像??蝬剝??
        # .sort_values(key=lambda s: s.abs(), ascending=False)嚗??抒?撠澆之撠?摨???????敺菜???
        coef_df = coefs.rename("coefficient").reset_index().rename(columns={"index": "feature"})
        # coefficient嚗???靽
        coef_df.loc[len(coef_df)] = {"feature": "intercept", "coefficient": float(model.intercept_.ravel()[0])}
        # model.intercept_嚗??扳芋???芾???ravel()[0]嚗?撟喳??洵銝?潘??虜?芣?銝????
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}_coefficients.csv"
        coef_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[Day7] ?⊥?頛詨蝺找??賂?{model_name}):{e}")
        """ ?交芋???舀 coef_ ??intercept_(靘?璅寞芋??嚗停?歲?航炊??
        蝔?銝?銝剜嚗??箄郎??"""

# 頛詨璅寞芋?敺?
def export_tree_feature_importance(model, feature_names: list[str], out_dir: Path, model_name: str):
    # 摰儔銝?撘?export_tree_feature_importance
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return # 憒?霈銝Logistic Regression瘝惇?改?撠梁??return 頝喲?
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        # 撱箇? pandas.Series嚗??詨澆???閬漲??銝????敺萄?蝔?
        # sort_values(ascending=False)嚗憭批撠?摨?霈????敺菜??其???
        out_dir.mkdir(parents=True, exist_ok=True) 
        # 蝣箔?頛詨鞈?憭曉??剁?mkdir(..., exist_ok=True)
        out_path = out_dir / f"{model_name}_feature_importance.csv"
        fi.rename("importance").reset_index().rename(columns={"index": "feature"}).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
    except Exception as e:
        print(f"[Day7] ?⊥?頛詨?孵噩??摨佗?{model_name}):{e}")
    """ 蝭拚?孵噩嚗??啣鈭敺菔甜?餅?憭改??臭誑瘙箏?靽???扎?
        閫??璅∪?嚗鼠?拚??銵犖?∠?閫?芋?蜓閬?寞??芯?霈靘?瑯?
        ?航?????敺???輸遢 CSV ?恍璇?(feature importance ranking) """

# 銝餅?蝔?
def run_day7(
    input_csv_path: str | Path = INPUT_CSV_PATH,
    target_column: str = TARGET_COLUMN_NAME,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """?瑁? Day 7:?游?憭芋??蝯曹?閮毀??隡啗??Ｗ?”/?勗?/????""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 霈鞈?
    df = pd.read_csv(input_csv_path)
    print(f"[Day7] 霈?伐?{input_csv_path} shape={df.shape}")

    # ?格?甈炎?亥?蝪∪摰寥嚗 Day4 撌脰????ㄐ?虜 OK嚗?
    check_target_column(df, target_column)
    if df[target_column].dtype == object:
        mapper = {"YES":1,"NO":0,"Y":1,"N":0,"TRUE":1,"FALSE":0,"??:1,"??:0,"1":1,"0":0,"1.0":1,"0.0":0}
        df[target_column] = df[target_column].astype(str).str.strip().str.upper().map(mapper)

    X, y = prepare_features_and_target(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split_simple(X, y, test_size, random_state)
    
    """  X_train, y_train ?其?閮毀璅∪?嚗?
         X_test, y_test 靽???蝯?隡堆?瘜??賢?嚗?"""

    # ?臬 sklearn ??詨???
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
        print("[Day7] 撠摰? scikit-learn,隢?摰?:python -m pip install scikit-learn")
        return ARTIFACTS_DIR
    """ 璅∪?瘙?Day7 ?喳???頛?蝔格?蝞?(Logistic?aive Bayes?邦璅∪??VM?NN 蝑???
        ??瘙??臬銝?渡???閰摯??嚗Ⅱ靽??芋??賜?詨?璅?靘?頛?
        摰寥嚗???scikit-learn 瘝?鋆?撠梁?交?蝷箔蝙?刻?摰?嚗?霈?撘援瞏啜?"""

    # ?舫嚗GBoost / LightGBM
    xgb = None
    lgbm = None
    # ??霈 xgb ??lgbm 閮剜? None嚗雿?閮凋?摮??
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=random_state
        )
    except Exception:
        pass
    """ ?岫?臬 XGBoost??
                n_estimators=300 ??璅寧??賊?嚗翮隞?活?賂?
                learning_rate=0.05 ??瘥活?湔?郊隡?撠郊?Ｚ粥嚗?頛帘摰?
                max_depth=4 ???格ㄤ璅寧??憭扳楛摨佗??踹??漲?砍?嚗?
                subsample=0.9 ??瘥活?冽???90% ?見?祈?蝺?
                colsample_bytree=0.9 ??瘥活?冽???90% ?敺?
                eval_metric="logloss" ??閰摯璅?閮剔撠?仃嚗?憿虜?剁?
                random_state=random_state ???箏??冽?蝔桀?嚗Ⅱ靽??? """
    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            random_state=random_state
        )
    except Exception:
        pass
    """ ?岫?臬 XGBoost??
                n_estimators=300 ??璅寧??賊?嚗翮隞?活?賂?
                learning_rate=0.05 ??瘥活?湔?郊隡?撠郊?Ｚ粥嚗?頛帘摰?
                max_depth=4 ???格ㄤ璅寧??憭扳楛摨佗??踹??漲?砍?嚗?
                subsample=0.9 ??瘥活?冽???90% ?見?祈?蝺?
                colsample_bytree=0.9 ??瘥活?冽???90% ?敺?
                eval_metric="logloss" ??閰摯璅?閮剔撠?仃嚗?憿虜?剁?
                random_state=random_state ???箏??冽?蝔桀?嚗Ⅱ靽??? """

    # 璅∪?皜
    models: list[tuple[str, object]] = [ # ?臭誑?刻艘???蝺氬?頛??芋??
        ("logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced")), # ?芸?隤踵憿甈?嚗???撟唾﹛璅?
        ("gaussian_naive_bayes", GaussianNB()), # ?渡?鞎??荔??身?孵噩蝚血?撣豢???
        ("decision_tree", DecisionTreeClassifier(random_state=random_state, class_weight="balanced")), # ?航撓???孵噩????(feature_importances_)
        ("random_forest", RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")), 
        # 憭ㄤ?冽?瘙箇?璅寞?蟡剁?瘥璉菜邦蝛拙嚗?憭?蝛抬?雿?蝺渲???
        ("gradient_boosting", GradientBoostingClassifier(random_state=random_state)), # 銝蝔格???嚗oosting嚗??郊靽格迤??璉菜邦?隤?
        ("svc_rbf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
        # ?詨??冽 RBF kernel,?臭誑?????扯??robability=True:霈芋?頛詨璈??閬?蝞?ROC?撩暺?憭扳?????Ｕ?
        ("knn", KNeighborsClassifier(n_neighbors=7)),
    ]
    if xgb is not None:
        models.append(("xgboost", xgb))
    if lgbm is not None:
        models.append(("lightgbm", lgbm))
        """  ?芸?餈凋誨閮毀??芋??
             蝯曹??園???accuracy?recision?ecall?1?OC AUC??
             憿?頛詨靽Logistic?敺菟?閬劫ree/XGB/LGBM??"""

    # ??蝛?撱箇????勗?瑼?
    with open(CLASSIFICATION_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Day7 Classification Reports\n")

    results = [] # 銝?征皜嚗靘??暹??芋??閰摯??嚗ccuracy?recision?ecall?1?oc_auc嚗?
    roc_sources = {} # 銝???賂?摮??芋?? ROC ?脩??函?璈?頛詨 (y_proba)嚗?敺憭芋??ROC ?脩?閬

    for model_name, model_obj in models:
        print(f"[Day7] 閮毀璅∪?嚗model_name}")
        model_obj.fit(X_train, y_train) # ?芋?閮毀鞈? ?砍?嚗飛蝧??賂?
        y_pred = model_obj.predict(X_test) # ?冽葫閰阡???皜穿?敺 0/1 憿蝯???
        """ 閮? accuracy / precision / recall / f1 / roc_auc
            頛詨 classification_report
            ?急毽瘛??OC ?脩?
            ?臬靽 / ?孵噩????"""

        # 璈?頛詨嚗?舀嚗?
        try:
            y_proba = model_obj.predict_proba(X_test)[:, 1] # ???蝑見?砍惇?潭迤憿 (label=1) ????
        except Exception:
            y_proba = None
        roc_sources[model_name] = y_proba # 蝯曹?摮 roc_sources嚗? ROC ?脩?瘥?雿輻

        # ??
        accuracy  = accuracy_score(y_test, y_pred) # 閮??孵? = (?葫甇?Ⅱ?見?祆) 繩 (?券璅???
        precision = precision_score(y_test, y_pred, zero_division=0) # 閮??孵? = ???繩 (???+ ???
        recall    = recall_score(y_test, y_pred, zero_division=0) # 閮??孵? = ???繩 (???+ ?????
        f1        = f1_score(y_test, y_pred, zero_division=0) # 閮??孵? = 2 ? (Precision ? Recall) 繩 (Precision + Recall)
        roc_auc   = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        """ roc_auc_score ?閬??撓?箝?y_proba)嚗??臬蝝?0/1 ?葫??
                ?澆?:0~1
                0.5 ???冽??葫
                0.7 蝞?亙?
                0.8 蝞憟?
                0.9 蝞蝘
 """
        results.append({ # ????隡唳?璅???銝???賂?? results list??
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }) # 頧? DataFrame嚗撓?箸? metrics.csv

        # 餈賢????勗?
        with open(CLASSIFICATION_REPORT_TXT, "a", encoding="utf-8") as f:
            f.write(f"\n=== {model_name} ===\n")
            f.write(classification_report(y_test, y_pred, digits=4))

        # 瘛瑟??拚 瘛瑟??拚?賡＊蝷箝P, FP, TN, FN??撣?瘥蝝?accuracy ?渡閫
        cm_path = CM_DIR / f"confusion_matrix_{model_name}.png"
        plot_confusion_matrix(y_test, y_pred, cm_path)

        # 蝺扳芋???? 閫????敺萄??葫敶梢?憭改?甇?鞎蔣?選?
        if model_name in {"logistic_regression"}:
            export_linear_coefficients(model_obj, X_train.columns.tolist(), COEF_CSV_DIR, model_name)
        """ 頛詨嚗??敺萇? 餈湔飛靽 (coefficient) + ?芾? (intercept) """

        # 璅寞芋?敺菟?閬漲: 撟怠??芋?靘??芯??孵噩?典??斗
        if model_name in {"decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm"}:
            export_tree_feature_importance(model_obj, X_train.columns.tolist(), FI_CSV_DIR, model_name)

        """  ?急毽瘛????閬死??TP / FP / TN / FN??
            頛詨 Logistic Regression ???????敺萄蔣?踵??
            頛詨 Tree-based 璅∪??敺菟?閬漲 ???鈭敺菜????"""

    # 摮?metrics & ROC嚗?璅∪?嚗?
    pd.DataFrame(results).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    plot_roc_multi(y_test, roc_sources, ROC_PNG)
    """ metrics.csv ??銵冽嚗靘?Excel / Pandas ??憭芋???蜀??
        roc_curve.png ??閬死??頛??芋??ROC ?脩???"""

    print(f"[Day7] ??摰?頛詨嚗n- {METRICS_CSV}\n- {CLASSIFICATION_REPORT_TXT}\n- {ROC_PNG}\n- {CM_DIR}/*.png\n- {COEF_CSV_DIR}嚗蝺扳芋??\n- {FI_CSV_DIR}嚗璅寞芋??")
    return ARTIFACTS_DIR


if __name__ == "__main__":
    run_day7()

