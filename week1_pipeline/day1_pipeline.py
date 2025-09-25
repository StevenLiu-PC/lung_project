# Day1 pipeline: 專案結構與讀寫流程
# 註解：僅新增說明，不影響程式邏輯


# Step 1: ?臬?閬?憟辣
import os               # ?其?撱箇?鞈?憭橘?os.makedirs嚗?
import pandas as pd     # ?其?霈????鞈?
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # ? lung_project ?寧??
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import csv_path, RAW_DIR, PROCESSED_DIR, RAW_OUT, PROC_OUT

def run_day1():
    print("ok: pandas ready")  # 蝣箄? pandas 撌脩??臭誑甇?虜雿輻

    # 霈瑼??
    if not Path(csv_path).exists():
        print(f"[?航炊] ?曆??啗撓?交?嚗csv_path}")
        sys.exit(1)

    # Step 3: 頛鞈?
    df = pd.read_csv(csv_path)  # 霈??CSV 瑼?嚗??喃???DataFrame嚗??”嚗?
    print("Shape (raw, cols):", df.shape)          # ?亦? (?, 甈)
    print("Columns:", df.columns.tolist())         # ???甈?
    print(df.head(3))                              # 瑼Ｘ??3 ???迤蝣箸?

    # Step 4: 蝯曹?甈??澆?嚗蝛箇??憭批神?征?賣摨?嚗?
    # .strip() ?駁??蝛箇??
    # .upper() ?????之撖恬??踹?憭批?撖急毽瘛?
    # .replace(" ", "_") ?征?賣??蝺??踹?銋??函?撘?雿??粹??
    df.columns = [M.strip().upper().replace(" ", "_") for M in df.columns]

    # Step 5: 瑼Ｘ甈??豢?行迤蝣綽??乩?蝣箏??府??16 甈?
    expected_cols = 16
    if len(df.columns) != expected_cols:
        print(f"[霅血?] 甈??訾?蝚佗?霈??{len(df.columns)} 甈?? {expected_cols}嚗?)
        print(f"撖阡?甈?嚗df.columns.tolist()}")

    # Step 6: 撠璅?雿?LUNG_CANCER 頧??詨?嚗ES=1, NO=0嚗?
    if "LUNG_CANCER" not in df.columns:
        raise ValueError(f"?曆???LUNG_CANCER 甈?嚗祕??雿?{df.columns.tolist()}")

    df["LUNG_CANCER"] = (
        df["LUNG_CANCER"]
          .astype(str)               # 靽???潮?臬?銝脣????踹??毽?摮?蝛箏潘?
          .str.upper()               # 頧?憭批神嚗??'yes' / 'Yes' / 'YES' 鋡怎???潘?
          .str.strip()               # ?餅???蝛箇嚗???CSV ?臬??憭蝛箸嚗?
          .map({"YES": 1, "NO": 0})  # 撱箇???嚗ES??, NO??
    )

    # Step 6.1: 瑼Ｘ?臬?瘜?????
    if df["LUNG_CANCER"].isna().any():    # 瑼Ｘ LUNG_CANCER 甈?鋆⊥?血??函征??
        bad_rows = df.loc[df["LUNG_CANCER"].isna(), ["LUNG_CANCER"]]
        raise ValueError(f"LUNG_CANCER ?箇??YES/NO ?潘?隢?皜??見?穿?\n{bad_rows.head()}")

    # Step 7: 撱箇?頛詨鞈?憭橘?憒?銝??典停撱箇?嚗?
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 7.1: 頛詨 RAW CSV
    df.to_csv(RAW_OUT, index=False, encoding="utf-8-sig")  # -sig ?舫??Excel 鈭Ⅳ
    print(f"Wrote RAW CSV: {RAW_OUT}")

    # Step 8: ?? PROCESSED ?嚗宏??ID 甈?嚗???嚗?
    drop_candidates = [c for c in ["ID", "PATIENT_ID"] if c in df.columns]
    if drop_candidates:
        df_processed = df.drop(columns=drop_candidates)
        print(f"?芷鈭?雿? {drop_candidates}")
    else:
        df_processed = df.copy()
        print("瘝?甈??閬??)

    print(df_processed.head())  # 憿舐內??5 蝑???

    # ??Day1 頛詨嚗摰???Day2/Day3 ????
    df_processed.to_csv(PROC_OUT, index=False, encoding="utf-8-sig")
    print(f"Wrote PROCESSED CSV: {PROC_OUT}")

    # Step 9: 蝪∪瑼Ｘ蝯?
    print("Processed shape:", df_processed.shape)
    print("Target distribution (LUNG_CANCER):")
    print(df_processed["LUNG_CANCER"].value_counts(dropna=False))
if __name__ == "__main__":
    run_day1()

