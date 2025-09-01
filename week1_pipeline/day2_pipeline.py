import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent  # 指到 lung_project 根目錄
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from project_config import PROC_OUT, OUTPUT_DAY2

def run_day2():
    # 讀取 Day1 輸出（相當於你原本上半段跑完的 df_processed）
    if not Path(PROC_OUT).exists():
        raise FileNotFoundError(f"[錯誤] 找不到 Day1 輸出：{PROC_OUT}，請先執行 Day1。")
    df_processed = pd.read_csv(PROC_OUT, encoding="utf-8-sig")

    cat_cols = df_processed.select_dtypes(include="object").columns.tolist()
    yesno_cols = []  # 收集真正的 YES/NO 欄位名稱

    for col in cat_cols:
        # 將此欄位標準化（轉字串、大寫、去前後空白），用於檢查值域
        vals = df_processed[col].astype(str).str.upper().str.strip()
        uniq = set(vals.dropna().unique())  # 去除缺失值後取得唯一值集合
        if uniq.issubset({"YES", "NO"}):   # 只有 YES/NO 才視為二元欄位
            yesno_cols.append(col)

    print(f"[Day2] 偵測到 YES/NO 欄位：{yesno_cols if yesno_cols else '（無）'}")

    # 對辨識出的 YES/NO 欄位做映射；轉為 float 方便後續統計與缺失值處理
    for col in yesno_cols:
        df_processed[col] = (
            df_processed[col]
            .astype(str).str.upper().str.strip()   # 與檢查時一致的正規化
            .map({"YES": 1, "NO": 0})              # YES→1, NO→0
            .astype("float")                        # 轉數值，方便 mean() 等操作
        )

    # ------------------------------------------------
    # Step 11：檢查缺失值情況
    # 顯示各欄位 NaN 的數量，快速掌握需要處理的欄位
    # ------------------------------------------------
    print("\n=== 缺失值統計（各欄位 NaN 筆數）===")
    na_counts = df_processed.isna().sum()
    print(na_counts[na_counts > 0].sort_values(ascending=False) if na_counts.any() else "無缺失值")

    # ------------------------------------------------
    # Step 12：缺失值處理（示範）
    # 原則：
    # - 數值欄位（int/float）→ 以平均值填補
    # - 仍為字串的欄位（object）→ 以眾數填補
    # 注意：正式專案可依情境改為中位數、固定值或進階填補法
    # ------------------------------------------------
    # 12.1 數值欄位（同時涵蓋 int 與 float）
    num_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_processed[col].isna().any():
            mean_val = df_processed[col].mean()
            df_processed[col] = df_processed[col].fillna(mean_val)

    # 12.2 仍為字串的欄位（若存在）
    obj_cols = df_processed.select_dtypes(include="object").columns
    for col in obj_cols:
        if df_processed[col].isna().any():
            mode_val = df_processed[col].mode(dropna=True)
            if not mode_val.empty:
                df_processed[col] = df_processed[col].fillna(mode_val.iloc[0])

    print("\n[Day2] 缺失值處理完成")

    # ------------------------------------------------
    # Step 13：資料分佈檢視
    # - 若存在 AGE，輸出其描述統計
    # - 對 LUNG_CANCER 顯示計數與百分比分佈，便於檢查類別不平衡
    # ------------------------------------------------
    if "AGE" in df_processed.columns:
        print("\n=== 年齡分佈（AGE.describe）===")
        print(df_processed["AGE"].describe())

    print("\n=== LUNG_CANCER 分佈（計數 / 百分比）===")
    if "LUNG_CANCER" in df_processed.columns:
        counts = df_processed["LUNG_CANCER"].value_counts(dropna=False)
        perc = (df_processed["LUNG_CANCER"].value_counts(normalize=True, dropna=False) * 100).round(2)
        print("計數：")
        print(counts)
        print("\n百分比（%）：")
        print(perc)
    else:
        print("找不到 LUNG_CANCER 欄位")

    # ------------------------------------------------
    # Step 14：輸出 Day 2 處理後的檔案
    # 與 Day 1 分檔存放，避免覆蓋；使用 UTF-8 確保中文不亂碼
    # ------------------------------------------------
    OUTPUT_DAY2.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(OUTPUT_DAY2, index=False, encoding="utf-8-sig")
    print(f"✅ Day 2 CSV 已儲存：{OUTPUT_DAY2}")

    # ------------------------------------------------
    # Step 15 分組折線圖
    if "AGE" in df_processed.columns:

        # 分組，例如每 5 歲一組
        bin_size = 5
        age_bins = pd.cut(df_processed["AGE"],
                          bins=range(int(df_processed["AGE"].min()), int(df_processed["AGE"].max()) + bin_size, bin_size))

        # 計算每組筆數
        age_group_counts = age_bins.value_counts().sort_index()

        # 用中位數當作 x 軸座標
        bin_midpoints = [interval.mid for interval in age_group_counts.index]

        plt.plot(bin_midpoints, age_group_counts.values, marker='o', linewidth=2, color="teal")
        plt.title("Age Distribution (Grouped Count)")
        plt.xlabel("Age Group Midpoint")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    run_day2()
