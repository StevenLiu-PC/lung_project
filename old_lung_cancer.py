# Step 1: 匯入需要的套件
import os               # 用來建立資料夾（os.makedirs）
import pandas as pd     # 用來讀取與處理資料
import numpy as np
import sys
from pathlib import Path

print("ok: pandas ready")  # 確認 pandas 已經可以正常使用

# Step 2: 設定檔案路徑（你的肺癌問卷 CSV 檔案）
# r"..." 代表原始字串（raw string），可以避免 Windows 路徑的 \ 被當成跳脫字元。
csv_path = r"D:\Microsoft VS Code\javacsript\Lung cancer\survey lung cancer.csv"

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
os.makedirs("data_lung/raw", exist_ok=True)
os.makedirs("data_lung/processed", exist_ok=True)

# Step 7.1: 輸出 RAW CSV
raw_out = "data_lung/raw/survey_lung_raw.csv"
df.to_csv(raw_out, index=False, encoding="utf-8-sig")  # -sig 可避免 Excel 亂碼
print(f"Wrote RAW CSV: {raw_out}")

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
proc_out = "data_lung/processed/day1_processed.csv"
df_processed.to_csv(proc_out, index=False, encoding="utf-8-sig")
print(f"Wrote PROCESSED CSV: {proc_out}")

# Step 9: 簡單檢查結果
print("Processed shape:", df_processed.shape)
print("Target distribution (LUNG_CANCER):")
print(df_processed["LUNG_CANCER"].value_counts(dropna=False))


# ------------------------------------------------
# Step 10：批次轉換 YES/NO → 1/0（僅轉真正 YES/NO 的欄位）
# 做法：
# 1) 先找出所有字串欄位（object）
# 2) 檢查每欄的「去空白、轉大寫」後的唯一值集合
# 3) 若值域僅包含 YES/NO（或含空值），才做映射成 1/0
# 這樣可避免把像 CITY/SEX/NAME 等非 YES/NO 欄位錯誤轉換為 NaN
# ------------------------------------------------
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
output_day2 = Path("data_lung/processed/day2_processed.csv")
df_processed.to_csv(output_day2, index=False, encoding="utf-8-sig")
print(f"✅ Day 2 CSV 已儲存：{output_day2}")

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

INPUT_CSV = Path("data_lung/processed/survey_lung_day2.csv")  # Day2 的實際輸出
OUTPUT_CSV = Path("data_lung/processed/day3_features.csv")    # 建議也放 processed
YES_RATIO_PLOT = Path("data_lung/plots/yes_ratio_top20.png")  # 圖片集中到 plots
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
YES_RATIO_PLOT.parent.mkdir(parents=True, exist_ok=True)
TOPK = 20  # 繪圖時取 YES 比例最高的前 K 題
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
    return len(unexpected) == 0            #檢查是否有 YES/NO 以外的值
                                        # 等於 0 代表乾淨

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
    #for c in df.columns：就是逐一把欄位名稱拿出來
    #if pattern.search(str(c))
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
                return lower_map[c.lower()]   # 再逐一比對候選名稱的小寫型態，看有沒有出現在 lower_map
        return None     # 如果完全找不到，回傳 None

    h_col = find_col(h_candidates)   #呼叫含式
    w_col = find_col(w_candidates)

    if h_col and w_col:
        # 如果同時有找到身高欄位 (h_col) 與體重欄位 (w_col)，才繼續執行
    # Python 判斷式：h_col 與 w_col 都不是 None 時為 True
        h_cm = pd.to_numeric(df[h_col], errors="coerce")
        # pd.to_numeric：把字串或其他型態轉成數字
        #把體重/身高欄位轉換成數值 (公分／公斤)，無法轉換的值會變成 NaN
        w_kg = pd.to_numeric(df[w_col], errors="coerce")
        h_m = h_cm / 100.0      # 將身高由公分 (cm) 轉換成公尺 (m)
        bmi = w_kg / (h_m ** 2) # 計算 BMI：體重 (公斤) / 身高平方 (公尺^2)
        df["BMI"] = bmi       # 在 DataFrame 新增一個 "BMI" 欄位，存放計算結果
        df["Obesity"] = df["BMI"].apply(lambda x: "肥胖" if pd.notna(x) and x >= 27 else ("正常" if pd.notna(x) else pd.NA))
    return df
     # 在 DataFrame 新增一個 "Obesity" 欄位，標記肥胖狀態
    # .apply(lambda x: ...)：對 "BMI" 欄位的每個值做判斷
    # 條件解釋：用pd.notna(x)語法
    #   - 如果 x 不是 NaN 且 >= 27 → 標記為 "肥胖" 套用.apply(lambda x: ...)規則
    #   - 如果 x 不是 NaN 且 < 27 → 標記為 "正常"  套用.apply(lambda x: ...)規則
    #   - 如果 x 是 NaN → 標記為缺失值 pd.NA    pd.notna(x)
    # 回傳更新後的 DataFrame，包含新計算的 BMI 與 Obesity 欄位
def main(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not Path(input_csv).exists():
        print(f"[錯誤] 找不到輸入檔：{input_csv}")
        sys.exit(1)
    df = pd.read_csv(input_csv)
#  main() 就是程式的「入口函式」(entry point)。
#  任何資料處理、分析、存檔流程，都會在這裡被串接起來。
#  確保輸入存在
#  一開始先檢查輸入檔 (input_csv) 有沒有存在，避免程式執行到一半才報錯。
# 這是防呆設計，讓錯誤能早點被抓到。
#  讀進同一份資料 (df)
#  pd.read_csv(input_csv) 讀進 DataFrame，之後所有步驟都基於這個 df。
# 一致性：所有處理 (清理、加 BMI、算 YES 比例、繪圖) 都針對同一份 df。
# 可控性：你清楚知道資料從哪裡來、要存去哪裡。

# # 可追蹤性：輸入 → 處理 → 輸出，流程清楚明瞭。
    

    #  自動辨識並轉換 YES/NO 欄位為 1/0
    yesno_cols = []    #先建立一個空清單 yesno_cols。
    for col in df.columns:   #對所有欄位名稱跑 for col in df.columns
        if is_yesno_column(df[col]):  #用 is_yesno_column() 檢查欄位內容是不是 YES/NO 型的題目。
            yesno_cols.append(col)    #如果是 → 把欄位名稱加入 yesno_cols。

    # 2 轉換
    df_bin = df.copy()
    for col in yesno_cols:
        df_bin[col] = coerce_yesno_to_binary(df_bin[col])

    # 3 吸菸分數 / 風險分群
    smoke_cols = find_smoking_columns(df_bin)
    usable_smoke_cols = []
    for c in smoke_cols:
        tmp = pd.to_numeric(df_bin[c], errors="coerce")
        if tmp.notna().sum() > 0:
            df_bin[c] = tmp
            usable_smoke_cols.append(c)

    if usable_smoke_cols:
        df_bin["smoke_score"] = df_bin[usable_smoke_cols].sum(axis=1, skipna=True)
        df_bin["smoke_risk"] = df_bin["smoke_score"].apply(
            lambda s: "高風險" if pd.notna(s) and s >= 2 else ("低風險" if pd.notna(s) else pd.NA)
        )
    else:
        df_bin["smoke_score"] = pd.NA
        df_bin["smoke_risk"] = pd.NA

    # 4) BMI 與肥胖標記（若有欄位才會新增）
    df_bin = add_bmi_and_obesity(df_bin)

    # 5) YES 比例（每題平均，因 1=YES, 0=NO）→ 僅針對成功轉成 0/1 的欄位
    yes_ratio = pd.Series(dtype=float)
    if yesno_cols:
        yes_ratio = df_bin[yesno_cols].mean(numeric_only=True).sort_values(ascending=False)

    # 6) 繪圖：YES 比例 TopK（有資料才畫）
    if yes_ratio.empty:
        print("[提醒] 無可計算的 YES/NO 欄位，略過繪圖。")
    else:
        try:
            topk = yes_ratio.head(min(TOPK, len(yes_ratio)))
            plt.figure(figsize=(max(8, len(topk) * 0.45), 4.8))
            topk.plot(kind="bar")
            plt.title(f"Top {len(topk)} YES 比例題目")
            plt.ylabel("YES 比例")
            plt.tight_layout()
            plt.savefig(YES_RATIO_PLOT, dpi=150)
            plt.close()
            print(f"[完成] 已輸出柱狀圖：{YES_RATIO_PLOT}")
        except Exception as e:
            print(f"[警告] 繪圖失敗：{e}")

    # 7) 輸出強化後資料（無論是否繪圖都會執行）
    df_bin.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[完成] 已輸出 Day3 強化後資料：{output_csv}")

    # 8) 終端簡報
    print("\n=== Day3 摘要 ===")
    print(f"- 偵測並轉換 YES/NO 欄位數：{len(yesno_cols)}")
    print(f"- 偵測吸菸相關欄位：{usable_smoke_cols if usable_smoke_cols else '（未發現或不可用）'}")
    if "BMI" in df_bin.columns:
        valid_bmi = df_bin['BMI'].notna().sum()
        print(f"- 已新增 BMI 與 Obesity（有效 BMI 筆數：{valid_bmi}）")
    if not yes_ratio.empty:
        print(f"- YES 比例最高的 5 題：")
        print(yes_ratio.head(5).round(3))

if __name__ == "__main__":
    # 支援從命令列指定輸入/輸出：
    # python day3_pipeline.py input.csv output.csv
    if len(sys.argv) >= 2:
        inpath = Path(sys.argv[1])
        outpath = Path(sys.argv[2]) if len(sys.argv) >= 3 else OUTPUT_CSV
        main(inpath, outpath)
    else:
        main()