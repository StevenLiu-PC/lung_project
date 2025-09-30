# Step 1: 匯入需要的套件
import os               # 用來建立資料夾（os.makedirs）
import pandas as pd     # 用來讀取與處理資料
import numpy as np
import sys
import re
from pathlib import Path
from project_config import RAW_DIR, PROCESSED_DIR

# ---------- Day1 會用到的工具 ----------

def ensure_dirs():
    """建立輸出資料夾（如果不存在就建立）"""
    # Step 7: 建立輸出資料夾（如果不存在就建立）
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_csv(csv_path: Path) -> pd.DataFrame:
    """讀檔防呆 + 載入資料"""
    # 讀檔防呆
    if not Path(csv_path).exists():
        print(f"[錯誤] 找不到輸入檔：{csv_path}")
        sys.exit(1)
    # Step 3: 載入資料
    df = pd.read_csv(csv_path)  # 讀取 CSV 檔案，回傳一個 DataFrame（資料表）。
    print("Shape (raw, cols):", df.shape)          # 查看 (列數, 欄數)
    print("Columns:", df.columns.tolist())         # 列出原始欄名
    print(df.head(3))                              # 檢查前 3 列資料正確性
    return df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: 統一欄名格式（去空白、轉大寫、空白改底線）
    .strip() 去除前後空白。
    .upper() 把欄名轉成大寫，避免大小寫混淆。
    .replace(" ", "_") 把空白改成底線，避免之後用程式操作時出錯。
    """
    df = df.copy()
    df.columns = [M.strip().upper().replace(" ", "_") for M in df.columns]
    return df

def validate_columns(df: pd.DataFrame, expected_cols: int | None):
    """Step 5: 檢查欄位數是否正確（若你確定應該是 16 欄）"""
    if expected_cols is not None and len(df.columns) != expected_cols:
        print(f"[警告] 欄位數不符，讀到 {len(df.columns)} 欄（應為 {expected_cols}）。")
        print(f"實際欄位：{df.columns.tolist()}")

def map_lung_cancer(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: 將目標欄位 LUNG_CANCER 轉成數字（YES=1, NO=0）"""
    if "LUNG_CANCER" not in df.columns:
        raise ValueError(f"找不到 LUNG_CANCER 欄位，實際欄位：{df.columns.tolist()}")

    df = df.copy()
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
    return df

def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Step 8: 處理 PROCESSED 版本（移除 ID 欄位，如果有）"""
    drop_candidates = [c for c in ["ID", "PATIENT_ID"] if c in df.columns]
    if drop_candidates:
        df_processed = df.drop(columns=drop_candidates)
        print(f"刪除了欄位: {drop_candidates}")
    else:
        df_processed = df.copy()
        print("沒有欄位需要刪除")
    print(df_processed.head())  # 顯示前 5 筆資料
    return df_processed

def save_csv(df: pd.DataFrame, out_path: Path, label: str):
    """統一輸出（使用 utf-8-sig，Excel 不亂碼）"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {label}: {out_path}")

# ========= 以下為「保留你的註解」的 Day2/Day3 工具；Day1 先不使用 =========

def normalize_yesno_series(s: pd.Series) -> pd.Series:
    """清理資料"""
    if s.dtype == object:
        return s.astype(str).str.strip().str.upper()
    return s
    # .astype(str) → 把值轉成字串，避免混入數字出錯
    # .str.strip() → 去掉前後空白
    # .str.upper() → 全部轉大寫，避免大小寫不一致

def is_yesno_column(s: pd.Series) -> bool:
    """
    判斷是否為 YES/NO 欄位：
    - 去除缺值後的唯一值皆屬於 {YES, NO, 1, 0, Y, N, 是, 否}
    """
    allowed = {"YES", "NO", "Y", "N", "1", "0", "TRUE", "FALSE", "是", "否"}
    vals = pd.Series(s.dropna().astype(str).str.strip().str.upper().unique())
    # 空欄或只有一種值也視為可轉（常見於全 NO/全 YES）
    if len(vals) <= 1:     #檢查這個欄位有幾種不同的答案
        return True
    unexpected = [v for v in vals if v not in allowed]
    return len(unexpected) == 0            #檢查是否有 YES/NO 以外的值（等於 0 代表乾淨）

def coerce_yesno_to_binary(s: pd.Series) -> pd.Series:
    """將 YES/NO 等對映為 1/0；無法辨識者設為 NaN"""
    mapping = {
        "YES": 1, "Y": 1, "TRUE": 1, "是": 1, "1": 1,
        "NO": 0, "N": 0, "FALSE": 0, "否": 0, "0": 0,
    }
    s_norm = normalize_yesno_series(s)
    return s_norm.map(mapping)

def find_smoking_columns(df: pd.DataFrame) -> list:
    """
    以關鍵字自動偵測吸菸相關欄位（適用英文欄名；若你有中文欄名可自行擴充）
    """
    keywords = [
        r"smok", r"cig", r"tobacco", r"pack", r"nicotin", r"pipe", r"quit",
        r"secondhand", r"passive", r"vape", r"e-?cig", r"hookah"
    ]
    pattern = re.compile("|".join(keywords), flags=re.IGNORECASE) #把清單裡的字樣用|分隔出來 暫時忽略大小寫
    smoke_cols = [c for c in df.columns if pattern.search(str(c))]
    #for c in df.columns：就是逐一把欄位名稱拿出來；if pattern.search(str(c))
    return smoke_cols   #把找到的「吸菸相關欄位名稱清單」回傳

def add_bmi_and_obesity(df: pd.DataFrame) -> pd.DataFrame:
    """
    若存在 Height(公分) 與 Weight(公斤) 欄位，計算 BMI 與 Obesity(>=27)。
    欄名不分大小寫；會嘗試常見變體。
    """
    # 定義可能的身高/體重欄位名稱候選清單（中英文、大寫小寫）
    # h_candidates 變數是一個 list，裡面列出多種可能的身高欄位名稱
    h_candidates = ["height", "身高", "Height", "HEIGHT"]
    w_candidates = ["weight", "體重", "Weight", "WEIGHT"]

    def find_col(cands):
        for c in cands:  #逐一檢查候選名稱，如果直接存在於 df.columns，就回傳這個欄位
            if c in df.columns:
                return c
        # 寬鬆匹配：忽略大小寫
        lower_map = {col.lower(): col for col in df.columns}
        for c in cands:
            if c.lower() in lower_map:         # 如果沒找到，就把所有欄位名稱轉成小寫，做一個 mapping
                return lower_map[c.lower()]    # 再逐一比對候選名稱的小寫型態，看有沒有出現在 lower_map
        return None     # 如果完全找不到，回傳 None

    h_col = find_col(h_candidates)   #呼叫含式
    w_col = find_col(w_candidates)

    if h_col and w_col:
        # 如果同時有找到身高欄位 (h_col) 與體重欄位 (w_col)，才繼續執行
        # Python 判斷式：h_col 與 w_col 都不是 None 時為 True
        h_cm = pd.to_numeric(df[h_col], errors="coerce")
        # pd.to_numeric：把字串或其他型態轉成數字；無法轉換的值會變成 NaN
        w_kg = pd.to_numeric(df[w_col], errors="coerce")
        h_m = h_cm / 100.0      # 將身高由公分 (cm) 轉換成公尺 (m)
        bmi = w_kg / (h_m ** 2) # 計算 BMI：體重 (公斤) / 身高平方 (公尺^2)
        df["BMI"] = bmi         # 在 DataFrame 新增一個 "BMI" 欄位，存放計算結果
        df["Obesity"] = df["BMI"].apply(
            lambda x: "肥胖" if pd.notna(x) and x >= 27 else ("正常" if pd.notna(x) else pd.NA)
        )
    return df
    # 在 DataFrame 新增一個 "Obesity" 欄位，標記肥胖狀態
    # .apply(lambda x: ...)：對 "BMI" 欄位的每個值做判斷
    # 條件解釋：用 pd.notna(x) 語法
    #   - 如果 x 不是 NaN 且 >= 27 → 標記為 "肥胖"
    #   - 如果 x 不是 NaN 且 < 27 → 標記為 "正常"
    #   - 如果 x 是 NaN → 標記為缺失值 pd.NA
