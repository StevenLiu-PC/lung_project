
# Step 1: 匯入需要的套件
import os               # 用來建立資料夾（os.makedirs）
import pandas as pd     # 用來讀取與處理資料
import numpy as np
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # 指到 lung_project 根目錄
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import csv_path, RAW_DIR, PROCESSED_DIR, RAW_OUT, PROC_OUT

def run_day1():
    print("ok: pandas ready")  # 確認 pandas 已經可以正常使用

    # 讀檔防呆
    if not Path(csv_path).exists():
        print(f"[錯誤] 找不到輸入檔：{csv_path}")
        sys.exit(1)

    # Step 3: 載入資料
    df = pd.read_csv(csv_path)  # 讀取 CSV 檔案，回傳一個 DataFrame（資料表）。
    print("Shape (raw, cols):", df.shape)          # 查看 (列數, 欄數)
    print("Columns:", df.columns.tolist())         # 列出原始欄名
    print(df.head(3))                              # 檢查前 3 列資料正確性

    # Step 4: 統一欄名格式（去空白、轉大寫、空白改底線）
    # .strip() 去除前後空白。
    # .upper() 把欄名轉成大寫，避免大小寫混淆。
    # .replace(" ", "_") 把空白改成底線，避免之後用程式操作時出錯。
    df.columns = [M.strip().upper().replace(" ", "_") for M in df.columns]

    # Step 5: 檢查欄位數是否正確（若你確定應該是 16 欄）
    expected_cols = 16
    if len(df.columns) != expected_cols:
        print(f"[警告] 欄位數不符，讀到 {len(df.columns)} 欄（應為 {expected_cols}）。")
        print(f"實際欄位：{df.columns.tolist()}")

    # Step 6: 將目標欄位 LUNG_CANCER 轉成數字（YES=1, NO=0）
    if "LUNG_CANCER" not in df.columns:
        raise ValueError(f"找不到 LUNG_CANCER 欄位，實際欄位：{df.columns.tolist()}")

    df["LUNG_CANCER"] = (
        df["LUNG_CANCER"]
          .astype(str)               # 保證所有值都是字串型態（避免有混雜數字或空值）
          .str.upper()               # 轉成大寫（避免 'yes' / 'Yes' / 'YES' 被當成不同值）
          .str.strip()               # 去掉前後空白（有時 CSV 匯出時會多出空格）
          .map({"YES": 1, "NO": 0})  # 建立映射：YES→1, NO→0
    )

    # Step 6.1: 檢查是否有無法轉換的值
    if df["LUNG_CANCER"].isna().any():    # 檢查 LUNG_CANCER 欄位裡是否存在空值
        bad_rows = df.loc[df["LUNG_CANCER"].isna(), ["LUNG_CANCER"]]
        raise ValueError(f"LUNG_CANCER 出現非 YES/NO 值，請先清理。樣本：\n{bad_rows.head()}")

    # Step 7: 建立輸出資料夾（如果不存在就建立）
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 7.1: 輸出 RAW CSV
    df.to_csv(RAW_OUT, index=False, encoding="utf-8-sig")  # -sig 可避免 Excel 亂碼
    print(f"Wrote RAW CSV: {RAW_OUT}")

    # Step 8: 處理 PROCESSED 版本（移除 ID 欄位，如果有）
    drop_candidates = [c for c in ["ID", "PATIENT_ID"] if c in df.columns]
    if drop_candidates:
        df_processed = df.drop(columns=drop_candidates)
        print(f"刪除了欄位: {drop_candidates}")
    else:
        df_processed = df.copy()
        print("沒有欄位需要刪除")

    print(df_processed.head())  # 顯示前 5 筆資料

    # ✅ Day1 輸出（固定檔名，Day2/Day3 會讀這個）
    df_processed.to_csv(PROC_OUT, index=False, encoding="utf-8-sig")
    print(f"Wrote PROCESSED CSV: {PROC_OUT}")

    # Step 9: 簡單檢查結果
    print("Processed shape:", df_processed.shape)
    print("Target distribution (LUNG_CANCER):")
    print(df_processed["LUNG_CANCER"].value_counts(dropna=False))
if __name__ == "__main__":
    run_day1()
