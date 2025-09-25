# Day5 pipeline: 基礎模型與指標
# 註解：僅新增說明，不影響程式邏輯

# --- 蝣箔???import 撠??寧? project_config.py ---
import sys
from pathlib import Path

# week1_pipeline/xxx_pipeline.py ??撠???(lung_project)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------

"""
Day 5: Baseline 撱箸芋撉冽
- 霈??Day4 皜?敺???(day4_cleaned.csv)
- 蝣箄?/鋆??格?甈?
- 憿甈?one-hot嚗??斤璅?嚗?
- 閮毀/皜祈岫??
- ?拙?baseline: Logistic Regression / GaussianNB
- ????銵刻撓?綽?metrics.csv?lassification_report.txt?OC?毽瘛???logreg) 靽銵?
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 銝剜?摮?嚗?賂?
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]   #matplotlib ??閮剖???凝頠迤暺?
plt.rcParams["axes.unicode_minus"] = False    #霈漣璅遘銝? 鞎??臭誑甇?虜憿舐內??

# === 頝臬?閮剖?嚗?蝙??project_config嚗??身嚗?===
try:
    from project_config import OUTPUT_CSV_DAY4 as INPUT_CSV  # day4_cleaned.csv
except Exception:
    INPUT_CSV = Path("data_lung/processed/day4_cleaned.csv")

# ?格?甈?蝔梧??交???project_config ?? TARGET_COL ???剁?
try:
    from project_config import TARGET_COL as DEFAULT_TARGET
except Exception:
    DEFAULT_TARGET = "LUNG_CANCER"

# ???舫嚗蜓?菔身摰??亙 project_config ?? KEY_COL ??KEY_COLS ????
try:
    from project_config import KEY_COLS as DEFAULT_KEYS   # ?刻皜嚗? ["PATIENT_ID"]
except Exception:
    try:
        from project_config import KEY_COL as _SINGLE_KEY  # ???桐? key
        DEFAULT_KEYS = [_SINGLE_KEY]
    except Exception:
        DEFAULT_KEYS = None  # ?亦閮剖?嚗?敺??典虜閬?璉蜓?萄?朣?

ARTIFACTS = Path("artifacts_day5")   #摰儔 Day5 ?撓?箄??冗????
METRICS_CSV = ARTIFACTS / "metrics.csv"     #?芋???湧???
CLF_REPORT_TXT = ARTIFACTS / "classification_report.txt"  #摰??憿??precision/recall/f1/?舀嚗?
ROC_PNG = ARTIFACTS / "roc_curve.png"   #ROC ?脩???
CM_DIR = ARTIFACTS / "confusion_matrices"  #摮??芋??瘛瑟??拚??
COEF_CSV = ARTIFACTS / "logreg_coefficients.csv"   #Logistic Regression ?敺萎??賂??急頝?嚗靘輯圾霈?孵噩敶梢??
""" ??/ 蝯楝敺?渲死?祕?神瑼?嚗?撘??遣鞈?憭?頝臬??舐撠??瑁???撌乩??桅?嚗銝?鞈?憭曉銵?頛詨撠望??賢撠?雿蔭 """

# ??撠極?瘀??銝駁甈???DEFAULT_KEYS 瘝?靘?撠梁撣貉??嚗?
def _pick_keys(df: pd.DataFrame) -> list[str] | None:
    """
    ??敺?獢身摰?撣貉??銝哨??曉?舐?蜓?菜?雿??券?質?摮??df)??
      - ? None 銵函內?曆??啣??拐蜓?蛛?銋????摨血?朣??梯郎????
    """ 
    #銝駁撠??臭?霅????犖????????臭犖嚗??? A ??蝐日???B嚗?
    # 1) 撠?閮剖??芸?
    if DEFAULT_KEYS and all(k in df.columns for k in DEFAULT_KEYS): #??撠?閮剖?鋆⊥?瘝????蜓??DEFAULT_KEYS
        return list(DEFAULT_KEYS)  #?銝隞賣鞎??撠??孵???DEFAULT_KEYS

    # 2 撣貉??嚗?摨?閰佗?
    candidates = [
        ["PATIENT_ID"],
        ["ID"],
        ["PATIENT_ID", "VISIT_DATE"],  # 蝭?銴???
    ]
    for keys in candidates:
        if all(k in df.columns for k in keys):
            return keys
    return None    #銝憚?賣銝撠勗???None???嚗?撐銵冽???蝣箇?銝駁?舐

# 撠極?瘀?瑼Ｘ?格?甈?
def _check_target(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"[Day5] ?格?甈?`{target}` 銝??冽鞈?銝准?)
    nuniq = df[target].nunique(dropna=True)
    if nuniq < 2:
        raise ValueError(f"[Day5] ?格?甈?`{target}` ?芣? {nuniq} ???伐??⊥??脰?????)
""" 甈?摮?改??Ⅱ隤?target ??雿?憒?LUNG_CANCER???刻??”鋆～??停?湔?梢?踹?敺撏拇蔑??
    憿?賊?:  nunique(dropna=True) 閮???蝛箏潛?銝?憿?詻?憿撠??拍車靘? 0 ??1??
              憒? < 2隞?”?湔??賭??舐征?賣?雿?典?銝??0 ??1瘝齒瘜?蝺游?憿?隞亦??颱??胯?"""
"""  -------------------------- """
# 撠極?瘀???Day4 瘝??格?甈??岫??Day3 鋆?
def _maybe_restore_target_from_day3(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target in df.columns:
        return df
       #憒? Day4 ?”?潸ㄐ撌脩??璅?嚗?憒?LUNG_CANCER),撠曹??湔??銝???隞仿??銴????暹?鞈?撘?
    # ????閰虫蜓?萄?朣??摰嚗?
    keys = _pick_keys(df)
    if keys is not None:
        try:
            # ?芾? Day3 ??keys + target嚗??賭蔔嚗?
            usecols = keys + [target]
            day3 = pd.read_csv("data_lung/processed/day3_features.csv", usecols=usecols)

            #  銝撠?撽?嚗?蜓?菟銝???
            if day3.duplicated(keys).any():
                dups = day3[day3.duplicated(keys, keep=False)][keys].head(5)
                raise ValueError(f"[Day5] Day3 銝駁??嚗瘜?撠?撠??見?穿?\n{dups}")

            if df.duplicated(keys).any():
                dups = df[df.duplicated(keys, keep=False)][keys].head(5)
                raise ValueError(f"[Day5] Day4 銝駁??嚗瘜?撠?撠??見?穿?\n{dups}")
            #??霅??蝑???銝駁 ????銝駁?賭?敺?銴????銝駁??Day3 ??蝐斗?蝣箄票??Day4
            # 隞?Day4 ?箔蜓嚗椰??撠?璅惜
            merged = df.merge(day3, on=keys, how="left", validate="one_to_one")
            #?銝駁?椰???撥?嗡?撠?撽?嚗票銝撠?NaN嚗票?舐?亙?胯?
            # ??瑼Ｘ?臬??銝璅惜
            miss = merged[target].isna().sum()
            if miss:
                bad = merged.loc[merged[target].isna(), keys].head(10)
                raise ValueError(
                    f"[Day5] ?望? {miss} 蝑??瘜銝駁撠 `{target}`??瑼Ｘ keys={keys} ?臬銝?氬n璅?嚗n{bad}"
                )

            print(f"[Day5] ??隞乩蜓??{keys} ??敺?Day3 撠?銝西???`{target}`??)
            return merged

        except FileNotFoundError:
            print("[Day5] ?曆???Day3 瑼?嚗?券摨血?朣?閰西???)
        except Exception as e:
            print(f"[Day5] 銝駁撠?鋆?蝐文仃???寧?瑕漲撠?嚗e}")

    # 嚗???嗆活嚗摨血?朣??芣??典銵函??詨??其??氬?鞈???銝?湔?????
    try:
        day3 = pd.read_csv("data_lung/processed/day3_features.csv", usecols=[target])
        if len(day3) == len(df):
            df[target] = day3[target].values
            print(f"[Day5] 瘜冽?嚗{target}` 銝 Day4 頛詨嚗歇敺?Day3 隞乓摨血?朣???)
        else:
            print("[Day5] Day3 ??Day4 蝑銝?嚗瘜??璅???)
    except Exception as e:
        print(f"[Day5] ?⊥?敺?Day3 鋆??格?甈?`{target}`嚗e}")
    return df
""" 1.?炎?伐?憒? Day4 ??DataFrame df 撌脩???target 甈?撠梁?亙??喉?銝?嚗?
    2.?血?嚗???銝駁撠??? Day3 鋆?嚗?摰嚗??嚗?
       - 銝撠?撽?嚗?蜓?萎?敺?銴?
       - 撌阡??蔥?撩?潭炎??
    3.?亦銝駁?蜓?萄?朣仃?????摨血?朣??伐?憸券頛?嚗??券摨衣??嚗?"""

# 撠極?瘀??孵噩??????格?甈?

def _prepare_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """蝣箔? X ?函?詨潦隞? object 甈?雿輻 pandas.get_dummies 頧 one-hot??格?甈?""
    # ??格?甈??踹??璅? one-hot ??
    obj_cols = [c for c in df.select_dtypes(include="object").columns.tolist() if c != target]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True, dummy_na=True)
        print(f"[Day5] 撌?one-hot 蝺函Ⅳ甈?嚗obj_cols}")

    # ? y 敺??? X ???刻???簣inf?aN??嚗? y ?舀?蝐文???X ?舐敺萇??
    y = df[target]
    X = df.drop(columns=[target])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    #??隞颱???之/?⊿?撠??潘?np.inf, -np.inf嚗??撩??NaN嚗?????NaN 鋆? 0

    return X, y

# ???璈????嚗?瘥?????蝺湧?嚗rain嚗??葫閰阡?嚗est嚗?
def _train_test_split(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    n = len(X)  #?見?祆嚗?撟曉?鞈?嚗?
    idx = np.arange(n) #撱箔??揣撘??[0, 1, 2, ..., n-1]嚗?銝銝靘???
    rng = np.random.default_rng(seed=random_state) #撱箇?鈭?Ｙ??剁?憛?箏?蝔桀?嚗Ⅱ靽???
    rng.shuffle(idx)  #?冽???蝝Ｗ?
    cut = int((1 - test_size) * n)   #蝞閮毀??暺?憒?n=100, test_size=0.2 ??cut=80??
    tr_idx, te_idx = idx[:cut], idx[cut:]   #??80% ?嗉?蝺渡揣撘? 20% ?嗆葫閰衣揣撘?
    return X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx] #?券?蝝Ｗ??餅??????嚗_train, X_test, y_train, y_test
    #璅∪?閰摯 = AI ??函恣?扳??嗚?蝣箔?摰漱?箇?蝑????舫???
# ??ROC嚗?璅∪?嚗?
def _plot_roc(y_test, y_proba_dict: dict[str, np.ndarray], out_path: Path):
    plt.figure()
    try:  ## ?岫?怠?
        from sklearn.metrics import roc_curve, auc as _auc # 閮?AUC
        plotted = False
        for name, prob in y_proba_dict.items():
            if prob is None:  #憒??芋??蝯行?璈?嚗停銝??怠???ROC
                continue
            fpr, tpr, _ = roc_curve(y_test, prob)  #蝞閰脫芋?? FPR ??TPR
            plt.plot(fpr, tpr, label=f"{name} (AUC={_auc(fpr, tpr):.3f})")
            #fpr (False Positive Rate) = ??抒?   tpr (True Positive Rate) = ??抒?
            plotted = True #蝣箄??喳??恍?銝璇?
        if plotted:
            plt.plot([0, 1], [0, 1], "--") #??蝺?箸?蝺??其?瘥?璅∪??????冽?憟?敺椰銝? (0,0) ?啣銝? (1,1) ??閫?
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (Day5 Baselines)")
            plt.legend() #?怠?靘?Legend嚗?憿舐內瘥??脩?撠??芸芋??
            plt.tight_layout() #?芸?隤踵?嚗??鋡急?????
            plt.savefig(out_path, dpi=150) #摮???out_path嚗?憒?roc_curve.png嚗pi=150 隞?”閫??摨佗??詨?頞???皜
        else:
            print("[Day5] ?∪?冽??撓?綽??仿? ROC 蝜芸???)
    finally:  # ?∟????嚗?敺?摰?????
       
        plt.close()


# ?急毽瘛??

def _plot_confusion_matrix(y_test, y_pred, out_path: Path):
    from sklearn.metrics import confusion_matrix
    import itertools   #?臬 瘛瑟??拚撌亙嚗? itertools嚗靘艘??鈭雁摨扳?嚗?
    cm = confusion_matrix(y_test, y_pred) #瘛瑟??拚 (Confusion Matrix)???臭???2x2 ?”??
    #TN (True Negative): ?祕??0嚗芋????0        FP (False Positive): ?祕??0嚗芋? 1嚗炊?梧?
    #FN (False Negative): ?祕??1嚗芋? 0嚗??梧?  TP (True Positive): ?祕??1嚗芋????1
    plt.figure()
    plt.imshow(cm, interpolation="nearest")  #??????脫憛???
    plt.title("Confusion Matrix")
    plt.colorbar()  #??憿?餃漲璇?憿瘛望滓隞?”?詨?憭批?
    tick_marks = np.arange(cm.shape[0])  #閮剖? X? 頠詨摨佗?璅內?箝? / 1??
    plt.xticks(tick_marks, tick_marks); plt.yticks(tick_marks, tick_marks)
    # 璅酉?詨?
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): #?券?餈游???瘥?摮神?冽憛葉憭?
        plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.ylabel("True label"); plt.xlabel("Predicted label")  #Y 頠?= ?祕?潘?X 頠?= 璅∪??葫??
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)  #撱箄??冗嚗???瘝遣嚗?
    plt.savefig(out_path, dpi=150)                      #????瑼??雿蔭 out_path
    plt.close()
    """  ?芋???葫蝯?嚗??銝撘萸? vs. ???扯”嚗蒂?怠??摮絲靘?"""
# 銝餅?蝔?
def run_day5(
    input_csv: str | Path = INPUT_CSV,
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """?瑁? Day 5 baseline:Logistic / Naive Bayes,頛詨????銵具?""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    print(f"[Day5] 霈?伐?{input_csv} shape={df.shape}")

    # ??Day4 瘝??格?甈???閰血? Day3 鋆?嚗? ??銝駁撠?嚗?
    df = _maybe_restore_target_from_day3(df, target)

    # ?亦璅?隞?摮/??銝憿嚗?亙?臭誑??靽株???
    _check_target(df, target)

    # ?亦璅?隞摮葡嚗扔撠?嚗ay4 ?虜撌脫?澆?嚗??岫頧?0/1
    if df[target].dtype == object:
        mapper = {"YES":1,"NO":0,"Y":1,"N":0,"TRUE":1,"FALSE":0,"??:1,"??:0,"1":1,"0":0,"1.0":1,"0.0":0}
        df[target] = df[target].astype(str).str.strip().str.upper().map(mapper)

    X, y = _prepare_features(df, target)
    X_train, X_test, y_train, y_test = _train_test_split(X, y, test_size, random_state)

    # 頛 sklearn
    try:
        from sklearn.linear_model import LogisticRegression    #?摩餈湔飛嚗虜?其?????憿?
        from sklearn.naive_bayes import GaussianNB              #擃鞎??臬?憿
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            classification_report
        )
        #accuracy_score: 皞Ⅱ??????靘??recision_score: 蝎曄Ⅱ???葫?粹?找葉??撠???賣改???
        #recall_score: ?砍???????賣找葉??撠◤?靘??1_score: 蝎曄Ⅱ???砍???撟唾﹛???
        #roc_auc_score: ROC ?脩?銝蝛?AUC嚗??亥? 1 頞末嚗?
        # classification_report: 銝隞賣???precision/recall/f1/?舀?賊?嚗?
    except Exception:
        print("[Day5] 撠摰? scikit-learn隢?摰?python -m pip install scikit-learn")
        return ARTIFACTS   #蝔??蝯?嚗??蝥芋??蝺渡???

    # 璅∪???
    models = [
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),#?憭? 1000 甈∟翮隞???踹????嗆????
                                                    #憒? 0/1 憿銝像銵∴???矽?湔???
        ("gnb", GaussianNB()), #Gaussian Naive Bayes
    ]

    results = []  #摮瘥芋??閰摯蝯?
    roc_sources = {}  #?其?摮OC ?脩??閬?璈??

    # ??蝛?撱箇????勗?瑼?
    with open(CLF_REPORT_TXT, "w", encoding="utf-8") as f: 
        f.write("Day5 Classification Reports\n")
    #??銝??摮? classification_report.txt嚗???瘥芋??閰喟敦???勗?摮脣
    for name, model in models:
        model.fit(X_train, y_train)   #?刻?蝺渲????蝺湔芋??
        y_pred = model.predict(X_test) # #閮毀摰芋??嚗皜祈岫鞈??颯??葫?_pred = 璅∪??葫??憿???0 ??1嚗?

        # 璈?頛詨嚗?舀嚗?
        try:
            y_proba = model.predict_proba(X_test)[:, 1]#predict_proba ?策雿甈?撅祆憿 0 ???惇?潮???1 ????
                                                        #[:, 1] ???洵鈭?嚗?撠望?惇??1 = ???璈???
        except Exception:
            y_proba = None   #憒?璅∪?銝?湔???撠望?摮?None
        roc_sources[name] = y_proba  #???芋??璈??摮摮 roc_sources 鋆∴?敺??ROC ?脩?閬
        #憒?璅∪?銝??predict_proba嚗停??胯?

        # ??
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)   #????????犖鋆∴?璅∪????啣?撠?
        f1   = f1_score(y_test, y_pred, zero_division=0)     #蝎曄Ⅱ???砍?????撟喳?
        auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan  #?OC ?脩?銝蝛?
                                    #y_proba = ?葫?????蹂???ROC ?脩?嚗??芋???舀璈?嚗UC = NaN??

        results.append({   #???芋????蝮曉???脖??之皜 results??敺?頛詨??metrics.csv??
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
        })

        # ?勗?嚗蕭?神?伐?
        with open(CLF_REPORT_TXT, "a", encoding="utf-8") as f:
            f.write(f"\n=== {name} ===\n")
            f.write(classification_report(y_test, y_pred, digits=4))
            #???芋??閰喟敦?勗? 餈賢? ??classification_report.txt

        # 瘛瑟??拚??瘥芋??摮?撘蛛?
        cm_path = CM_DIR / f"confusion_matrix_{name}.png"
        _plot_confusion_matrix(y_test, y_pred, cm_path)

        # 靽銵剁??? Logistic嚗?
        if name == "logreg":
            try:
                # 靽撅像??蝬哨?蝝Ｗ?撠??孵噩?迂嚗蒂靘?撠潭?摨?
                coefs = pd.Series(model.coef_.ravel(), index=X_train.columns) \
                          .sort_values(key=lambda s: s.abs(), ascending=False) #?敺萄?蝔梁雿揣撘?撠?瘥???
                                                                             #靘??撠澆之撠?摨?敶梢?之?????
                coef_df = coefs.rename("coef").reset_index().rename(columns={"index": "feature"})#?嫣噶頛詨??Excel / CSV嚗???撘萸??詨蔣?踹?銵冽??
                # 銋?銝頝?intercept嚗?
                coef_df.loc[len(coef_df)] = {"feature": "intercept", "coef": float(model.intercept_.ravel()[0])}
                #Logistic Regression 銝?敺萎??賂???銝?頝?(intercept)???貊?澆皞???
                coef_df.to_csv(COEF_CSV, index=False, encoding="utf-8-sig") #?遢靽銵刻撓?箸? CSV
            except Exception as e:
                print(f"[Day5] ?⊥?頛詨靽(Logistic):{e}") #憒?隞颱??唳?粹嚗?憒芋????coef_ 撅祆改?嚗停?啣?航炊閮嚗?銝?霈??撘援瞏?

    # 摮?metrics & ROC
    pd.DataFrame(results).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")#results嚗??Ｚ?瘥芋??蝝舐???璅??殷?accuracy/precision/recall/f1/roc_auc嚗?
    _plot_roc(y_test, roc_sources, ROC_PNG)#????怠??Ｗ神憟賜? _plot_roc嚗?憭芋?? ROC ?脩??怠??撘萄?銝蒂摮?

    print(f"[Day5] ??摰?頛詨嚗n- {METRICS_CSV}\n- {CLF_REPORT_TXT}\n- {ROC_PNG}\n- {CM_DIR}/*.png\n- {COEF_CSV}嚗 Logistic ??頛詨嚗?)
    return ARTIFACTS
    """ metrics.csv:   ?芋?擃?璅”
    classification_report.txt:  瘥芋?????勗?嚗瘥??亦? precision/recall/f1)
    roc_curve.png:  ROC ??
    confusion_matrices/*.png: 瘥芋??瘛瑟??拚??
    logreg_coefficients.csv: Logistic ?敺萎??賂??交?嚗?"""

if __name__ == "__main__":
    run_day5()

