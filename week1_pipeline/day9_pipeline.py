# Day9 pipeline: 指標彙整與文字報告
# 註解：僅新增說明，不影響程式邏輯

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 撠??寧?身摰?
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ?臬 Day4 頛詨??????
try:
    from project_config import OUTPUT_CSV_DAY4 as INPUT_CSV_PATH
except Exception:
    INPUT_CSV_PATH = Path("data_lung/processed/day4_cleaned.csv")

try:
    from project_config import TARGET_COL as TARGET_COLUMN_NAME
except Exception:
    TARGET_COLUMN_NAME = "LUNG_CANCER"

# Day9 ??頛詨頝臬?
ARTIFACTS_DIR = Path("artifacts_day9")
REPORT_TXT = ARTIFACTS_DIR / "final_model_report.txt"
ROC_PNG = ARTIFACTS_DIR / "final_model_roc.png"
CM_PNG = ARTIFACTS_DIR / "final_model_confusion_matrix.png"

BEST_MODEL_PATH = Path("artifacts_day8/best_model.pkl")

# 撌亙?賢?

def check_target_column(dataframe: pd.DataFrame, target_column: str) -> None:
    """蝣箄??格?甈??臬摮銝撠憿?""
    if target_column not in dataframe.columns:
        raise ValueError(f"[Day9] ?曆??啁璅?嚗target_column}")
    if dataframe[target_column].nunique(dropna=True) < 2:
        raise ValueError(f"[Day9] ?格?甈?`{target_column}` 憿銝雲嚗瘜?憿?)

def prepare_features_and_target(dataframe: pd.DataFrame, target_column: str):
    """???孵噩??蝐歹?銝西???one-hot / 蝻箏潦?""
    categorical_columns = [c for c in dataframe.select_dtypes(include="object").columns if c != target_column]
    if categorical_columns:
        dataframe = pd.get_dummies(
            dataframe,
            columns=categorical_columns,
            drop_first=True,
            dummy_na=True
        )
        print(f"[Day9] One-hot 蝺函Ⅳ甈?嚗categorical_columns}")

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y

def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    """蝜芾ˊ瘛瑟??拚??""
    from sklearn.metrics import confusion_matrix # ?芸?賢??折??閬??典??典??
    import itertools # 撟怠??冽摮葉??璅酉?詨?
    cm = confusion_matrix(y_true, y_pred) # 閮?瘛瑟??拚嚗?閮剜甇?????湔閮嚗?

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues) # ?典蔣?撘＊蝷箇??
    plt.title("Confusion Matrix (Day9)")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # ?冽??摮葉??箸??
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 蝣箔?頛詨鞈?憭曉???
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_roc_curve(y_true, y_proba, out_path: Path):
    """ roc_curve() ?閬?亦?撖行?蝐?(y_true) ??皜祆???(y_proba) """
    from sklearn.metrics import roc_curve, auc # ?臬閮? ROC ?脩???AUC ?極??
    fpr, tpr, _ = roc_curve(y_true, y_proba) 
    # auc() 閮? ROC ?脩?銝蝛?(Area Under Curve)嚗﹛?芋?儘?亥??
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.3f})") # ?怠 ROC ?脩?
    plt.plot([0, 1], [0, 1], "--", color="gray")  # ?怠撠?蝺?(?冽??葫?皞?)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Day9)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    """ ROC ?脩?嚗?璅∪??其???琿?潦?嚗??賣抒? (TPR) ???賣抒? (FPR) ?像銵～?
        AUC ??ROC ?脩?銝蝛??詨潔???0.5(鈭?嚗 1.0(摰???嚗??亥? 1 頞末??
        撠?蝺?銵函內?冽??葫?皞?(AUC=0.5)??"""
    

# 銝餅?蝔?

def run_day9(
    input_csv_path: str | Path = INPUT_CSV_PATH,
    target_column: str = TARGET_COLUMN_NAME,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Day9: 雿輻 Day8 ?曉??雿單芋???脰??蝯葫閰西?頛詨?勗???""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    #蝣箔? TIFACTS_DIR ???冗摮嚗???摮撠曹?頝臬遣韏瑚?嚗??歇蝬??剁?撠勗蕭?伐?銝??梢??

    # 霈鞈?
    df = pd.read_csv(input_csv_path)
    print(f"[Day9] 霈?伐?{input_csv_path}, shape={df.shape}")# - DataFrame ?耦? (df.shape)
    check_target_column(df, target_column)

    # 蝪∪頧? YES/NO ??1/0
    if df[target_column].dtype == object: #   # 憒??格?甈??舀?摮??伐?object嚗???摮葡頧???皞?
        mapper = {"YES": 1, "NO": 0, "Y": 1, "N": 0, "TRUE": 1, "FALSE": 0, "??: 1, "??: 0}
        df[target_column] = df[target_column].astype(str).str.strip().str.upper().map(mapper)

    # ???孵噩 X ??蝐?y嚗??? one-hot ?撩?潘?
    X, y = prepare_features_and_target(df, target_column)

    # train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    ) # ?????? 閮毀??(X_train, y_train) ??皜祈岫??(X_test, y_test)

    # 頛?雿單芋??
    import joblib
    if not BEST_MODEL_PATH.exists():
        print("[Day9] ?曆???Day8 ?雿單芋??隢??瑁? Day8??)
        return ARTIFACTS_DIR # 憒?瘝?璅∪?嚗停?湔蝯??賢?嚗??唾撓?箄??冗頝臬?
    best_model = joblib.load(BEST_MODEL_PATH) # 頛?雿單芋??Day8 摮?靘?璅∪?瑼?

    # ?葫
    y_pred = best_model.predict(X_test) # ?湔?Ｙ??葫璅惜嚗?/1嚗?
    try:
        y_proba = best_model.predict_proba(X_test)[:, 1] # [:, 1] 隞?”?惇?潭迤憿??璈?
    except Exception:
        y_proba = None
        # 憒?璅∪?銝??predict_proba嚗?憒?SVM 瘝???probability嚗?
        # 撠望? y_proba 閮剜? None嚗??撘??

    # ??
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
    acc = accuracy_score(y_test, y_pred) # ?湧?蝑?瘥???
    prec = precision_score(y_test, y_pred, zero_division=0) # ????撠撠?
    rec = recall_score(y_test, y_pred, zero_division=0) # ?????啣?撠?
    f1 = f1_score(y_test, y_pred, zero_division=0) # Precision ??Recall ??銵瘀??踹??芰?銝??
    auc_score = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan 
                                            #璅∪??湧?????0.5=鈭?嚗?=摰?????

    # 頛詨?勗?
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("Day9 Final Model Report\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
        f.write(f"ROC-AUC  : {auc_score:.4f}\n\n")
        f.write("=== Classification Report ===\n")
        f.write(classification_report(y_test, y_pred, digits=4))
              # ?批捆? precision/recall/f1/?舀摨?(support)嚗?雿??豢撘?

    # 瘛瑟??拚 & ROC
    plot_confusion_matrix(y_test, y_pred, CM_PNG)
    if y_proba is not None:
        plot_roc_curve(y_test, y_proba, ROC_PNG)

    print(f"[Day9] ??摰??蝯芋?葫閰佗?頛詨嚗n- {REPORT_TXT}\n- {CM_PNG}\n- {ROC_PNG}")
    return ARTIFACTS_DIR


if __name__ == "__main__":
    run_day9()

