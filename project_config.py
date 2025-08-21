from pathlib import Path

# Step 2: 設定檔案路徑（你的肺癌問卷 CSV 檔案）
# r"..." 代表原始字串（raw string），可以避免 Windows 路徑的 \ 被當成跳脫字元。
csv_path = Path(r"D:\Microsoft VS Code\javacsript\Lung cancer\survey lung cancer.csv")

# 輸出資料夾
RAW_DIR = Path("data_lung/raw")
PROCESSED_DIR = Path("data_lung/processed")
PLOTS_DIR = Path("data_lung/plots")

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
