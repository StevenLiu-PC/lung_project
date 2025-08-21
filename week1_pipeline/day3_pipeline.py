import sys
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

from project_config import INPUT_CSV_DAY3 as INPUT_CSV
from project_config import OUTPUT_CSV_DAY3 as OUTPUT_CSV
from project_config import YES_RATIO_PLOT, TOPK

# Day3 會用到的路徑準備
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
YES_RATIO_PLOT.parent.mkdir(parents=True, exist_ok=True)

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
