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
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]   #matplotlib 的預設字型改成「微軟正黑體
plt.rcParams["axes.unicode_minus"] = False    #讓座標軸上的 負號可以正常顯示。

# === 路徑設定（優先使用 project_config，否則用預設） ===
try:
    from project_config import OUTPUT_CSV_DAY4 as INPUT_CSV  # day4_cleaned.csv
except Exception:
    INPUT_CSV = Path("data_lung/processed/day4_cleaned.csv")

# 目標欄名稱（若有在 project_config 指定 TARGET_COL 會優先採用）
try:
    from project_config import TARGET_COL as DEFAULT_TARGET
except Exception:
    DEFAULT_TARGET = "LUNG_CANCER"

# ★（可選）主鍵設定：若在 project_config 提供 KEY_COL 或 KEY_COLS 會優先採用
try:
    from project_config import KEY_COLS as DEFAULT_KEYS   # 推薦清單，如 ["PATIENT_ID"]
except Exception:
    try:
        from project_config import KEY_COL as _SINGLE_KEY  # 舊式單一 key
        DEFAULT_KEYS = [_SINGLE_KEY]
    except Exception:
        DEFAULT_KEYS = None  # 若無設定，稍後會用常見候選或放棄主鍵對齊

ARTIFACTS = Path("artifacts_day5")   #定義 Day5 的輸出資料夾與檔名
METRICS_CSV = ARTIFACTS / "metrics.csv"     #各模型的整體指標
CLF_REPORT_TXT = ARTIFACTS / "classification_report.txt"  #完整的分類報告（precision/recall/f1/支援）
ROC_PNG = ARTIFACTS / "roc_curve.png"   #ROC 曲線圖
CM_DIR = ARTIFACTS / "confusion_matrices"  #存每個模型的混淆矩陣圖片
COEF_CSV = ARTIFACTS / "logreg_coefficients.csv"   #Logistic Regression 的特徵係數（含截距），方便解讀特徵影響力
""" 用 / 組路徑更直覺。實際寫檔前，程式會先建資料夾,路徑是相對你執行時的工作目錄；在不同資料夾執行，輸出就會落在對應位置 """

# ★ 小工具：挑選主鍵欄（若 DEFAULT_KEYS 沒提供，就用常見候選）
def _pick_keys(df: pd.DataFrame) -> list[str] | None:
    """
    ★ 從專案設定或常見候選中，找出可用的主鍵欄位（全部都要存在於 df)。
      - 回傳 None 表示找不到合適主鍵（之後會退回到「長度對齊」或報警告）。
    """ 
    #主鍵對齊是保證：同一個人的「整列」資料不會對錯人（不會把 A 的標籤配到 B）
    # 1) 專案設定優先
    if DEFAULT_KEYS and all(k in df.columns for k in DEFAULT_KEYS): #先看專案設定裡有沒有指定的主鍵 DEFAULT_KEYS
        return list(DEFAULT_KEYS)  #回傳一份拷貝避免不小心改到原本的 DEFAULT_KEYS

    # 2 常見名單（依序嘗試）
    candidates = [
        ["PATIENT_ID"],
        ["ID"],
        ["PATIENT_ID", "VISIT_DATE"],  # 範例複合鍵
    ]
    for keys in candidates:
        if all(k in df.columns for k in keys):
            return keys
    return None    #三輪都找不到就回傳 None。意思是：目前這張表沒有明確的主鍵可用

# 小工具：檢查目標欄
def _check_target(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"[Day5] 目標欄 `{target}` 不存在於資料中。")
    nuniq = df[target].nunique(dropna=True)
    if nuniq < 2:
        raise ValueError(f"[Day5] 目標欄 `{target}` 只有 {nuniq} 個類別，無法進行分類。")
""" 欄位存在性：先確認 target 這個欄位例如 LUNG_CANCER真的在資料表裡。沒有就直接報錯避免後面崩潰。
    類別數量:  nunique(dropna=True) 計算「非空值的不同類別數」。分類至少要兩種例如 0 和 1。
              如果 < 2代表整欄都不是空白欄位、全部同一個值 0 或 1沒辦法訓練分類器所以立刻丟錯。 """
"""  -------------------------- """
# 小工具：若 Day4 沒有目標欄，嘗試自 Day3 補回
def _maybe_restore_target_from_day3(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target in df.columns:
        return df
       #如果 Day4 的表格裡已經有目標欄（例如 LUNG_CANCER),就什直接原封不動回傳。可以避免重複補或把現有資料弄亂
    # ★ 先嘗試主鍵對齊（最安全）
    keys = _pick_keys(df)
    if keys is not None:
        try:
            # 只讀 Day3 的 keys + target（效能佳）
            usecols = keys + [target]
            day3 = pd.read_csv("data_lung/processed/day3_features.csv", usecols=usecols)

            #  一對一驗證：兩邊主鍵都不得重複
            if day3.duplicated(keys).any():
                dups = day3[day3.duplicated(keys, keep=False)][keys].head(5)
                raise ValueError(f"[Day5] Day3 主鍵重複，無法一對一對齊。樣本：\n{dups}")

            if df.duplicated(keys).any():
                dups = df[df.duplicated(keys, keep=False)][keys].head(5)
                raise ValueError(f"[Day5] Day4 主鍵重複，無法一對一對齊。樣本：\n{dups}")
            #先找「能識別同一筆資料」的主鍵 → 雙邊主鍵都不得重複 → 再用主鍵把 Day3 的標籤準確貼回 Day4
            # 以 Day4 為主，左連接對齊標籤
            merged = df.merge(day3, on=keys, how="left", validate="one_to_one")
            #先找主鍵→左連接→強制一對一驗證；貼不到就 NaN，貼錯直接報錯。
            # ★ 檢查是否有對不到標籤
            miss = merged[target].isna().sum()
            if miss:
                bad = merged.loc[merged[target].isna(), keys].head(10)
                raise ValueError(
                    f"[Day5] 共有 {miss} 筆資料無法由主鍵對到 `{target}`。請檢查 keys={keys} 是否一致。\n樣本：\n{bad}"
                )

            print(f"[Day5] ★ 以主鍵 {keys} 成功從 Day3 對齊並補回 `{target}`。")
            return merged

        except FileNotFoundError:
            print("[Day5] 找不到 Day3 檔案，改用長度對齊嘗試補回。")
        except Exception as e:
            print(f"[Day5] 主鍵對齊補標籤失敗，改用長度對齊：{e}")

    # （退而求其次）長度對齊：只有在兩表筆數完全一致、且資料順序一致時才安全
    try:
        day3 = pd.read_csv("data_lung/processed/day3_features.csv", usecols=[target])
        if len(day3) == len(df):
            df[target] = day3[target].values
            print(f"[Day5] 注意：`{target}` 不在 Day4 輸出，已從 Day3 以『長度對齊』補回。")
        else:
            print("[Day5] Day3 與 Day4 筆數不同，無法補回目標欄。")
    except Exception as e:
        print(f"[Day5] 無法從 Day3 補回目標欄 `{target}`：{e}")
    return df
""" 1.先檢查：如果 Day4 的 DataFrame df 已經有 target 欄，就直接回傳（不動）。
    2.否則：優先用「★主鍵對齊」從 Day3 補回（最安全），包含：
       - 一對一驗證（兩邊主鍵不得重複）
       - 左連接合併、缺值檢查
    3.若無主鍵或主鍵對齊失敗，再退回到『長度對齊』策略（風險較高，僅在長度相同時）。 """

# 小工具：特徵前處理（排除目標欄）

def _prepare_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """確保 X 全為數值。若仍有 object 欄，使用 pandas.get_dummies 轉為 one-hot排除目標欄。"""
    # 排除目標欄，避免把目標欄 one-hot 掉
    obj_cols = [c for c in df.select_dtypes(include="object").columns.tolist() if c != target]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True, dummy_na=True)
        print(f"[Day5] 已 one-hot 編碼欄位：{obj_cols}")

    # 取出 y 後，再對 X 做安全處理（±inf→NaN→0）  y 是標籤向量；X 是特徵矩陣
    y = df[target]
    X = df.drop(columns=[target])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    #先把任何「無限大/無限小」的值（np.inf, -np.inf）換成缺值 NaN；再把所有 NaN 補成 0

    return X, y

# 把資料隨機打散後，依比例切成「訓練集（train）」跟「測試集（test）
def _train_test_split(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    n = len(X)  #取樣本數（有幾列資料）
    idx = np.arange(n) #建一個索引陣列 [0, 1, 2, ..., n-1]，等一下拿來洗牌。
    rng = np.random.default_rng(seed=random_state) #建立亂數產生器，塞入固定種子，確保「可重現」
    rng.shuffle(idx)  #隨機打亂索引
    cut = int((1 - test_size) * n)   #算出訓練集切點。例如 n=100, test_size=0.2 → cut=80。
    tr_idx, te_idx = idx[:cut], idx[cut:]   #前 80% 當訓練索引、後 20% 當測試索引
    return X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx] #用這些索引去抓列，回傳四包：X_train, X_test, y_train, y_test
    #模型評估 = AI 的「內部管控機制」，確保它交出的答案真的可靠。
# 畫 ROC（多模型）
def _plot_roc(y_test, y_proba_dict: dict[str, np.ndarray], out_path: Path):
    plt.figure()
    try:  ## 嘗試畫圖
        from sklearn.metrics import roc_curve, auc as _auc # 計算AUC
        plotted = False
        for name, prob in y_proba_dict.items():
            if prob is None:  #如果這個模型沒給我機率，就不要畫它的 ROC
                continue
            fpr, tpr, _ = roc_curve(y_test, prob)  #算出該模型的 FPR 與 TPR
            plt.plot(fpr, tpr, label=f"{name} (AUC={_auc(fpr, tpr):.3f})")
            #fpr (False Positive Rate) = 假陽性率   tpr (True Positive Rate) = 真陽性率
            plotted = True #確認至少畫過一條線
        if plotted:
            plt.plot([0, 1], [0, 1], "--") #這條線是基準線，用來比較模型有沒有比隨機好 從左下角 (0,0) 到右上角 (1,1) 的對角線
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (Day5 Baselines)")
            plt.legend() #畫圖例（Legend），顯示每條曲線對應哪個模型
            plt.tight_layout() #自動調整版面，避免字被擠掉或重疊
            plt.savefig(out_path, dpi=150) #存檔到 out_path（例如 roc_curve.png）。dpi=150 代表解析度，數字越高圖越清晰
        else:
            print("[Day5] 無可用機率輸出，略過 ROC 繪圖。")
    finally:  # 無論有沒有錯，最後一定要關掉圖
       
        plt.close()


# 畫混淆矩陣

def _plot_confusion_matrix(y_test, y_pred, out_path: Path):
    from sklearn.metrics import confusion_matrix
    import itertools   #匯入 混淆矩陣工具，跟 itertools（用來迴圈跑二維座標）
    cm = confusion_matrix(y_test, y_pred) #混淆矩陣 (Confusion Matrix)。它是一個 2x2 的表格
    #TN (True Negative): 真實為 0，模型也判 0        FP (False Positive): 真實為 0，模型判 1（誤報）
    #FN (False Negative): 真實為 1，模型判 0（漏報）  TP (True Positive): 真實為 1，模型也判 1
    plt.figure()
    plt.imshow(cm, interpolation="nearest")  #把矩陣畫成「顏色方塊圖」
    plt.title("Confusion Matrix")
    plt.colorbar()  #加上顏色刻度條，顏色深淺代表數字大小
    tick_marks = np.arange(cm.shape[0])  #設定 X、Y 軸刻度，標示出「0 / 1」
    plt.xticks(tick_marks, tick_marks); plt.yticks(tick_marks, tick_marks)
    # 標註數字
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): #用雙迴圈把 每格的數字寫在方塊中央
        plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.ylabel("True label"); plt.xlabel("Predicted label")  #Y 軸 = 真實值，X 軸 = 模型預測值
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)  #建資料夾（如果還沒建）
    plt.savefig(out_path, dpi=150)                      #把圖片存檔到指定位置 out_path
    plt.close()
    """  把模型的預測結果，整理成一張「真 vs. 假」對照表，並畫出圖檔存起來。 """
# 主流程
def run_day5(
    input_csv: str | Path = INPUT_CSV,
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """執行 Day 5 baseline:Logistic / Naive Bayes,輸出指標與圖表。"""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    print(f"[Day5] 讀入：{input_csv} shape={df.shape}")

    # 若 Day4 沒有目標欄，先嘗試從 Day3 補回（★ 會優先用主鍵對齊）
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
        from sklearn.linear_model import LogisticRegression    #邏輯迴歸（常用來做二元分類）
        from sklearn.naive_bayes import GaussianNB              #高斯貝葉斯分類器
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            classification_report
        )
        #accuracy_score: 準確率（分對的比例）。precision_score: 精確率（預測為陽性中有多少是真的陽性）。
        #recall_score: 召回率（所有真陽性中有多少被抓出來）。f1_score: 精確率與召回率的平衡分數。
        #roc_auc_score: ROC 曲線下面積（AUC，越接近 1 越好）。
        # classification_report: 一份整合報告（precision/recall/f1/支援數量）
    except Exception:
        print("[Day5] 尚未安裝 scikit-learn請先安裝python -m pip install scikit-learn")
        return ARTIFACTS   #程式提早結束，避免後續模型訓練爆炸

    # 模型們
    models = [
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),#最多跑 1000 次迭代，避免「沒收斂」報錯
                                                    #如果 0/1 類別不平衡，會自動調整權重
        ("gnb", GaussianNB()), #Gaussian Naive Bayes
    ]

    results = []  #存放每個模型的評估結果
    roc_sources = {}  #用來存「ROC 曲線需要的機率分數

    # 先清空/建立分類報告檔
    with open(CLF_REPORT_TXT, "w", encoding="utf-8") as f: 
        f.write("Day5 Classification Reports\n")
    #先開一個文字檔 classification_report.txt，準備把每個模型的詳細分類報告存進去
    for name, model in models:
        model.fit(X_train, y_train)   #用訓練資料去「訓練模型」
        y_pred = model.predict(X_test) # #訓練完模型後，拿測試資料去「做預測」。y_pred = 模型預測的分類結果（0 或 1）

        # 機率輸出（若支援）
        try:
            y_proba = model.predict_proba(X_test)[:, 1]#predict_proba 會給你兩欄：屬於類別 0 的機率、屬於類別 1 的機率。
                                                        #[:, 1] → 取第二欄（也就是「屬於 1 = 有肺癌的機率」）
        except Exception:
            y_proba = None   #如果模型不支援機率，就會存 None
        roc_sources[name] = y_proba  #把每個模型的機率分數存到字典 roc_sources 裡，後面畫 ROC 曲線要用
        #如果模型不支援 predict_proba，就會報錯。

        # 指標
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)   #所有真的有癌症的人裡，模型有抓到多少
        f1   = f1_score(y_test, y_pred, zero_division=0)     #精確率和召回率的加權平均
        auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan  #「ROC 曲線下面積」
                                    #y_proba = 預測的機率，拿來畫 ROC 曲線；如果模型不支援機率，AUC = NaN。

        results.append({   #把每個模型的「成績單」存進一個大清單 results。最後會輸出成 metrics.csv。
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
            #把每個模型的詳細報告 追加 到 classification_report.txt

        # 混淆矩陣圖（每個模型各存一張）
        cm_path = CM_DIR / f"confusion_matrix_{name}.png"
        _plot_confusion_matrix(y_test, y_pred, cm_path)

        # 係數表（僅對 Logistic）
        if name == "logreg":
            try:
                # 係數展平成一維，索引對應特徵名稱，並依絕對值排序
                coefs = pd.Series(model.coef_.ravel(), index=X_train.columns) \
                          .sort_values(key=lambda s: s.abs(), ascending=False) #把特徵名稱當作索引，對應每個係數
                                                                             #依照「絕對值大小」排序，影響力大的排前面。
                coef_df = coefs.rename("coef").reset_index().rename(columns={"index": "feature"})#方便輸出成 Excel / CSV，變成一張「變數影響力表格」
                # 也附上截距（intercept）
                coef_df.loc[len(coef_df)] = {"feature": "intercept", "coef": float(model.intercept_.ravel()[0])}
                #Logistic Regression 不只有特徵係數，還有一個「截距 (intercept)」，相當於基準點。
                coef_df.to_csv(COEF_CSV, index=False, encoding="utf-8-sig") #把這份係數表輸出成 CSV
            except Exception as e:
                print(f"[Day5] 無法輸出係數(Logistic):{e}") #如果任何地方出錯（例如模型沒有 coef_ 屬性），就印出錯誤訊息，但不會讓整個程式崩潰

    # 存 metrics & ROC
    pd.DataFrame(results).to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")#results：前面跑每個模型時累積的指標清單（accuracy/precision/recall/f1/roc_auc）。
    _plot_roc(y_test, roc_sources, ROC_PNG)#這行會呼叫前面寫好的 _plot_roc，把多個模型的 ROC 曲線畫在同一張圖上並存檔

    print(f"[Day5] ✅ 完成輸出：\n- {METRICS_CSV}\n- {CLF_REPORT_TXT}\n- {ROC_PNG}\n- {CM_DIR}/*.png\n- {COEF_CSV}（若 Logistic 成功輸出）")
    return ARTIFACTS
    """ metrics.csv:   各模型整體指標表
    classification_report.txt:  每個模型的分類報告（含每類別的 precision/recall/f1)
    roc_curve.png:  ROC 圖
    confusion_matrices/*.png: 每個模型的混淆矩陣圖
    logreg_coefficients.csv: Logistic 的特徵係數（若有） """

if __name__ == "__main__":
    run_day5()
