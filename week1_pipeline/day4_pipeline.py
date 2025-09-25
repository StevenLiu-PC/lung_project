# Day4 pipeline: 嚴格目標欄標準化（strict）
# 註解：僅新增說明，不影響程式邏輯

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

# 銝剜?摮?嚗?賂?
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# === 頝臬?閮剖?嚗?? project_config嚗?==
from project_config import (
    OUTPUT_CSV_DAY3 as INPUT_CSV,   #  Day3 ?撓?箝靘 Day4 ?撓??
    OUTPUT_CSV_DAY4 as OUTPUT_MAIN, # Day4 銝餉撓??
    TARGET_COL as TARGET,           # ?格?甈?嚗?憒?LUNG_CANCER嚗?
    LABEL_MAP,                      # ?舫?蝵桃?璅惜撠銵?
)
OUTPUT_DIR = Path(OUTPUT_MAIN).parent
OUTPUT_SUM = OUTPUT_DIR / "day4_feature_summary.csv"

""" ??銝脫?雿?蝢姊ormalization?踹??憭批?撖急?蝛箇撠?斗銝??氬?"""
def normalize_str(s: pd.Series) -> pd.Series: 
    if s.dtype == object:      #?斗??甈銝 ??????object 憿?嚗?
        return s.astype(str).str.strip().str.upper() 
                    #銝恣??頛詨??"Yes"?? yes "??YES"嚗?敺絞銝?質???"YES"??
    return s    #憒?銝??甈?嚗?憒摮?嚗?撠勗?璅???喉?銝?隞颱?靽格

# ???芰?潔??祆?雿? YES/NO 頧?1/0嚗璅??西???嚗?
GENERAL_YESNO_MAP = {
    "YES":1,"NO":0,"Y":1,"N":0,"TRUE":1,"FALSE":0,"??:1,"??:0,"1":1,"0":0,"1.0":1,"0.0":0
}
GENERAL_ALLOWED = set(GENERAL_YESNO_MAP.keys())

# ??Parquet 摰撖怠嚗?撘?撠勗神嚗??停?芷?頝喲?
def _write_parquet_safe(df: pd.DataFrame, out_path: Path) -> None:
    try:
        df.to_parquet(out_path, index=False, engine="pyarrow")  # ??
        print(f"[Day4] 撌脰撓??Parquet嚗yarrow嚗?{out_path}")   # ??
        return
    except Exception:
        pass
    try:
        df.to_parquet(out_path, index=False, engine="fastparquet")  # ??
        print(f"[Day4] 撌脰撓??Parquet嚗astparquet嚗?{out_path}")  # ??
        return
    except Exception:
        print("[Day4] 頝喲? Parquet 頛詨嚗摰? pyarrow/fastparquet ?????舐??)  # ??
        # ??銝?靘?嚗?瘚?蝜潛?嚗SV 隞?頛詨嚗?

def _normalize_target_strict(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    ???湔???格?甈??芸 Day4 ??霅?Day5 ?舐嚗?
    瘚?嚗?
      1) ?格?甈??????餌征??頧之撖?
      2) ? project_config.LABEL_MAP[TARGET] ????0/1
      3) ?乩???NaN,?岫?摮??捆?胯?靘?鞈??嗅祕??1/2)
      4) ?湔瑼Ｘ嚗?敺? NaN???箔???
    """
    if TARGET not in df.columns:
        raise ValueError(f"[Day4] ?曆??啁璅? `{TARGET}`??)

    # 1) 摮葡甇????
    s_raw = df[TARGET]
    if s_raw.dtype == object:
        s_norm = normalize_str(s_raw)
    else:
        s_norm = s_raw.astype(str).str.strip().str.upper()

    # 2) 撠銵剁??臬 project_config.LABEL_MAP ?游?嚗?
    target_map = LABEL_MAP.get(TARGET, GENERAL_YESNO_MAP)
    s_map = s_norm.map(target_map)

    # 3) ?詨?鈭?摰寥嚗? 1/2??/1 銋?嚗?
    if s_map.notna().sum() == 0:
        s_num = pd.to_numeric(s_norm, errors="coerce")
        uniq = sorted(v for v in s_num.dropna().unique())
        if len(uniq) == 2:
            a, b = uniq
            # ?乩??臬?憟?{0,1}嚗停??撠澆? 0??憭批澆? 1
            map2 = {a:0, b:1} if set(uniq) != {0,1} else {0:0, 1:1}
            print(f"[Day4] ?菜葫?唳摮???蝐?{uniq} ???∠撠? {map2}")
            s_map = s_num.map(map2)

    # 4) ?湔瑼Ｘ
    na_cnt = int(s_map.isna().sum())
    nunique = s_map.nunique(dropna=True)
    if strict:
        if na_cnt > 0:
            bad = normalize_str(s_raw)[s_map.isna()].value_counts().head(10)
            raise ValueError(
                f"[Day4] ?格?甈?`{TARGET}` ??{na_cnt} 蝑瘜儘霅?NaN嚗?
                f"\n隢 project_config.LABEL_MAP['{TARGET}'] 鋆??扼?
                f"\n[?芾儘霅?Top10]\n{bad}"
            )
        if nunique < 2:
            raise ValueError(f"[Day4] ?格?甈?`{TARGET}` ????nunique={nunique}嚗?)
        df[TARGET] = s_map
    else:
        # ??潭芋撘?銝? NaN ??
        if na_cnt:
            df = df[s_map.notna()].copy()
            s_map = s_map.loc[df.index]
        if s_map.nunique(dropna=True) < 2:
            raise ValueError(f"[Day4] ?格?甈?`{TARGET}` 皜?敺?????)
        df[TARGET] = s_map

    return df

def run_day4(strict: bool = True):
    # 1) 霈瑼?
    # df = pd.read_csv(INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")              # ??蝯曹?霈瑼楊蝣潘??? BOM
    df.columns = [c.lstrip("\ufeff") for c in df.columns]          # ???餅?甈??? BOM嚗??迎?
    print("[Day4] 霈?伐?", df.shape)

    # ???啣?嚗炎?交?行隞質”?芣? 1 甈??踹??芣? target ?餅??孵噩嚗?
    if df.shape[1] <= 1:
        raise ValueError("[Day4] 頛詨鞈??芣? 1 甈?蝻箏??孵噩嚗?瑼Ｘ Day3 頛詨?批捆??)

    # ??1.1) ???璅?嚗??璅惜嚗??敺蛛?
    df = _normalize_target_strict(df, strict=strict)

    # 2) ?澆摮葡摰儔嚗??斤璅?嚗?
    for col in df.select_dtypes(include="object").columns: #?曉???摮?甈?
        if col == TARGET:
            continue
        df[col] = normalize_str(df[col])

    # 3) 撠ES/NO 頧???1/0嚗??斤璅?嚗璅歇??嚗?
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        if col == TARGET: 
            continue
        uniq = set(df[col].dropna().unique().tolist()) #??甈?銝剜??????潘??餅?蝻箏潘?
        if len(uniq) > 0 and uniq.issubset(GENERAL_ALLOWED): #蝣箄???甈????迂??嚗?頧?
            df[col] = df[col].map(GENERAL_YESNO_MAP)  #??YES/NO 頧? 1/0

    # 4) 蝻箏潸???Missing Value Imputation嚗??孵噩甈??格?甈?憛怨?嚗?
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
    for col in num_cols:
        if df[col].isna().any():  #????澆????芾???鋆⊥?銝?撩?潘?撠勗???True
            df[col] = df[col].fillna(df[col].mean()) #?撩?潸府甈‵?亙像??

    # 5) ?曉瘝?霈???雿???格?甈?
    """ ????雿?蝔???瑼Ｘ瘥?雿??曉甈???撠車?臭??澆??南an,
    憒?甈??芣? 0 ??1 蝔桀潘?隞?”?? 瘝?霈? """
    constant_cols = [c for c in df.columns if c != TARGET and df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        print("[Day4] 蝘駁?∟???雿?", constant_cols)
        for c in constant_cols:
            uniq_vals = df[c].unique().tolist() #??numpy array 頧? Python ??list 霈”?格憟賜?
            print(f"  - {c} (?臭??? {uniq_vals})")#(f"...{霈}...") 隤?蝟?
        df = df.drop(columns=constant_cols)  #敺?DataFrame 蝘駁??甈?
    else:
        print("[Day4] 瘝??菜葫?啁霈?甈?")

    # 6) 撱箇?頛詨鞈?憭?
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) #蝣箔?瑼?????
    # df.to_csv(OUTPUT_MAIN, index=False)  #??末???” df 摮?銝??CSV 瑼?瑼?頝臬?撠望 OUTPUT_MAIN
    df.to_csv(OUTPUT_MAIN, index=False, encoding="utf-8-sig", sep=",")  # ???箏? sep/蝺函Ⅳ嚗xcel ???ay10 蝛抵?
    _write_parquet_safe(df, Path(OUTPUT_MAIN).with_suffix(".parquet"))  # ???折鈭斗?嚗撘???歲??

    # 7) 甈???嚗???蝻箏潭/?臭??潭/蝭?嚗?
    summary = df.apply(lambda s: pd.Series({
        "dtype": s.dtype,
        "na_cnt": int(s.isna().sum()),
        "nunique": int(s.nunique(dropna=False)),
        "sample": s.dropna().head(3).tolist()  # ???啣?嚗?嗥?靘撓?箔?閬云??
    }))#撟急撘菔??”??隞賬?雿牧????瘥?甈????撩?潭?銝?潭?誑??撟曄?蝭?嚗?
    # summary.to_csv(OUTPUT_SUM)  #摮?銝??CSV 瑼?
    summary.to_csv(OUTPUT_SUM, encoding="utf-8-sig")               # ????銋 utf-8-sig嚗xcel 憿舐內銝?蝣?

    print("??摰??蜓頛詨嚗?, OUTPUT_MAIN)
    print("??摰???閬撓?綽?", OUTPUT_SUM)
    print("[Day4] ?蝯耦?嚗?, df.shape)
    return OUTPUT_MAIN

if __name__ == "__main__":
    run_day4(strict=True)

