# encoding: utf-8
# eol: LF
# summary: feat(day4): 嚴格目標欄正規化 + 缺值處理

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

# 中文字體（可選）
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# === 路徑設定（優先吃 project_config）===
from project_config import (
    OUTPUT_CSV_DAY3 as INPUT_CSV,   # ✅ Day3 的【輸出】拿來當 Day4 的輸入
    OUTPUT_CSV_DAY4 as OUTPUT_MAIN, # Day4 主輸出
    TARGET_COL as TARGET,           # 目標欄名（例如 LUNG_CANCER）
    LABEL_MAP,                      # 可配置的標籤對照表
)
OUTPUT_DIR = Path(OUTPUT_MAIN).parent
OUTPUT_SUM = OUTPUT_DIR / "day4_feature_summary.csv"

""" 把字串欄位定義normalization避免因為大小寫或空白導致判斷不一致。 """
def normalize_str(s: pd.Series) -> pd.Series: 
    if s.dtype == object:      #判斷這一欄是不是 文字型資料（object 類型）
        return s.astype(str).str.strip().str.upper() 
                    #不管原始輸入是 "Yes"、" yes "、"YES"，最後統一都變成 "YES"。
    return s    #如果不是文字欄位（例如數字型），就原樣回傳，不做任何修改

# ★ 只用於一般欄位的 YES/NO 轉 1/0（目標欄另行處理）
GENERAL_YESNO_MAP = {
    "YES":1,"NO":0,"Y":1,"N":0,"TRUE":1,"FALSE":0,"是":1,"否":0,"1":1,"0":0,"1.0":1,"0.0":0
}
GENERAL_ALLOWED = set(GENERAL_YESNO_MAP.keys())

def _normalize_target_strict(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    ★ 嚴格處理目標欄（只在 Day4 做、保證 Day5 可用）
    流程：
      1) 目標欄存在 → 去空白+轉大寫
      2) 先用 project_config.LABEL_MAP[TARGET] 映射到 0/1
      3) 若仍全 NaN，嘗試「數字二元容錯」（例如資料其實是 1/2）
      4) 嚴格檢查：不得有 NaN、且為二元
    """
    if TARGET not in df.columns:
        raise ValueError(f"[Day4] 找不到目標欄 `{TARGET}`。")

    # 1) 字串正規化
    s_raw = df[TARGET]
    if s_raw.dtype == object:
        s_norm = normalize_str(s_raw)
    else:
        s_norm = s_raw.astype(str).str.strip().str.upper()

    # 2) 對照表（可在 project_config.LABEL_MAP 擴充）
    target_map = LABEL_MAP.get(TARGET, GENERAL_YESNO_MAP)
    s_map = s_norm.map(target_map)

    # 3) 數字二元容錯（如 1/2、0/1 之類）
    if s_map.notna().sum() == 0:
        s_num = pd.to_numeric(s_norm, errors="coerce")
        uniq = sorted(v for v in s_num.dropna().unique())
        if len(uniq) == 2:
            a, b = uniq
            # 若不是剛好 {0,1}，就把較小值對 0、較大值對 1
            map2 = {a:0, b:1} if set(uniq) != {0,1} else {0:0, 1:1}
            print(f"[Day4] 偵測到數字二元標籤 {uniq} → 採用對映 {map2}")
            s_map = s_num.map(map2)

    # 4) 嚴格檢查
    na_cnt = int(s_map.isna().sum())
    nunique = s_map.nunique(dropna=True)
    if strict:
        if na_cnt > 0:
            bad = normalize_str(s_raw)[s_map.isna()].value_counts().head(10)
            raise ValueError(
                f"[Day4] 目標欄 `{TARGET}` 有 {na_cnt} 筆無法辨識（NaN）。"
                f"\n請在 project_config.LABEL_MAP['{TARGET}'] 補對照。"
                f"\n[未辨識值 Top10]\n{bad}"
            )
        if nunique < 2:
            raise ValueError(f"[Day4] 目標欄 `{TARGET}` 非二元（nunique={nunique}）。")
        df[TARGET] = s_map
    else:
        # 非嚴格模式（僅本機清資料用，不建議放公開版）：丟掉 NaN 列
        if na_cnt:
            df = df[s_map.notna()].copy()
            s_map = s_map.loc[df.index]
        if s_map.nunique(dropna=True) < 2:
            raise ValueError(f"[Day4] 目標欄 `{TARGET}` 清理後仍非二元。")
        df[TARGET] = s_map

    return df

def run_day4(strict: bool = True):
    # 1) 讀檔
    df = pd.read_csv(INPUT_CSV)
    print("[Day4] 讀入：", df.shape)

    # ★ 1.1) 先處理目標欄（只處理標籤，不動特徵）
    df = _normalize_target_strict(df, strict=strict)

    # 2) 呼叫字串定義（排除目標欄）
    for col in df.select_dtypes(include="object").columns: #找出所有文字型欄位
        if col == TARGET:
            continue
        df[col] = normalize_str(df[col])

    # 3) 將YES/NO 轉換成 1/0（排除目標欄；目標已處理）
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        if col == TARGET: 
            continue
        uniq = set(df[col].dropna().unique().tolist()) #取得欄位中所有不重複的值（去掉缺值）
        if len(uniq) > 0 and uniq.issubset(GENERAL_ALLOWED): #確認這一欄真的只包含允許集合，才轉換
            df[col] = df[col].map(GENERAL_YESNO_MAP)  #把 YES/NO 轉成 1/0

    # 4) 缺值處理 Missing Value Imputation（僅特徵欄，目標欄不填補）
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
    for col in num_cols:
        if df[col].isna().any():  #會產生布林值序列，只要這欄裡有一個缺值，就回傳 True
            df[col] = df[col].fillna(df[col].mean()) #把缺值該欄填入平均值

    # 5) 找出沒有變化的欄位（排除目標欄）
    """ 取出所有欄位名稱,逐一檢查每個欄位,找出欄位有多少種唯一值包含nan,
    如果欄位只有 0 或 1 種值，代表這欄 沒有變化 """
    constant_cols = [c for c in df.columns if c != TARGET and df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        print("[Day4] 移除無變化欄位：", constant_cols)
        for c in constant_cols:
            uniq_vals = df[c].unique().tolist() #把 numpy array 轉成 Python 的 list 讓表單更好看
            print(f"  - {c} (唯一值: {uniq_vals})")#(f"...{變數}...") 語法糖
        df = df.drop(columns=constant_cols)  #從 DataFrame 移除這些欄位
    else:
        print("[Day4] 沒有偵測到無變化欄位")

    # 6) 建立輸出資料夾
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) #確保檔案有存入 
    df.to_csv(OUTPUT_MAIN, index=False)  #把整理好的資料表 df 存成一個 CSV 檔，檔案路徑就是 OUTPUT_MAIN
                                         #不要另外存索引欄（index=False）

    # 7) 欄位摘要（型態/缺值數/唯一值數/範例）
    summary = df.apply(lambda s: pd.Series({
        "dtype": s.dtype,
        "na_cnt": int(s.isna().sum()),
        "nunique": int(s.nunique(dropna=False)),
        "sample": s.dropna().head(3).tolist()
    }))#幫整張資料表做一份『欄位說明書』，包含每一欄的型態、缺值數量、唯一值數量、以及前幾筆範例，
    summary.to_csv(OUTPUT_SUM)  #存成一個 CSV 檔

    print("✅ 完成。主輸出：", OUTPUT_MAIN)
    print("✅ 完成。摘要輸出：", OUTPUT_SUM)
    print("[Day4] 最終形狀：", df.shape)
    return OUTPUT_MAIN

if __name__ == "__main__":
    run_day4(strict=True)

