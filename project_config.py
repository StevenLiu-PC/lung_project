from pathlib import Path
import os
# Step 2: 設定檔案路徑（你的肺癌問卷 CSV 檔案）
# r"..." 代表原始字串（raw string），可以避免 Windows 路徑的 \ 被當成跳脫字元。
csv_path = Path(os.getenv("LUNG_CSV", r"D:\Microsoft VS Code\javacsript\Lung cancer\survey lung cancer.csv"))

# 輸出資料夾
RAW_DIR = Path("data_lung/raw")       #存放原始資料（raw）的資料夾
PROCESSED_DIR = Path("data_lung/processed")   #存放清理/轉換後資料（processed）的資料夾
PLOTS_DIR = Path("data_lung/plots")     #圖表輸出（plots）的資料夾

# Day1 固定輸出（Day2 會讀這個）
RAW_OUT = RAW_DIR / "survey_lung_raw.csv"
PROC_OUT = PROCESSED_DIR / "survey_lung_day1.csv"

# Day2 固定輸出（Day3 會讀這個）
OUTPUT_DAY2 = PROCESSED_DIR / "survey_lung_day2.csv"

# Day3 設定
INPUT_CSV_DAY3 = PROCESSED_DIR / "survey_lung_day2.csv"   # Day2 的輸出
OUTPUT_CSV_DAY3 = PROCESSED_DIR / "survey_lung_day3.csv"  # Day3 的輸出
YES_RATIO_PLOT = PLOTS_DIR / "yes_ratio_top20.png"        # 圖片集中到 plots
TOPK = 20  # 繪圖時取 YES 比例最高的前 K 題
# 目標欄（Day4 會嚴格正規化、Day5 依此訓練）
TARGET_COL = "LUNG_CANCER"

# 可擴充的標籤對照表（Day4 先用這個把標籤轉為 0/1）
LABEL_MAP = {
    "LUNG_CANCER": {
        # 英文常見寫法
        "YES":1, "NO":0, "Y":1, "N":0, "TRUE":1, "FALSE":0, "T":1, "F":0,
        "POSITIVE":1, "NEGATIVE":0, "PRESENT":1, "ABSENT":0,
        # 中文常見寫法
        "是":1, "否":0, "有":1, "無":0, "陽性":1, "陰性":0,
        # 數字字串
        "1":1, "0":0, "1.0":1, "0.0":0
    }
}

# Day4 主輸出（Day5 會讀這個）
OUTPUT_CSV_DAY4 = PROCESSED_DIR / "survey_lung_day4.csv"
# 也可以用既有命名風格：PROCESSED_DIR / "day4_cleaned.csv"

# Day5 產物資料夾（ROC、混淆矩陣、metrics 等）
ARTIFACTS_DAY5 = Path("artifacts_day5")

def ensure_dirs():
    for p in [RAW_DIR, PROCESSED_DIR, PLOTS_DIR, ARTIFACTS_DAY5]:
        p.mkdir(parents=True, exist_ok=True)
KEY_COLS = ["PATIENT_ID"]     # 或複合鍵：["PATIENT_ID", "VISIT_DATE"]