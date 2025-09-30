import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # lung_project 資料夾
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]  # 正黑體，避免中文字變口口口
plt.rcParams["axes.unicode_minus"] = False  # 避免負號變亂碼

# ★ 建議在 project_config.py 內，讓 INPUT_CSV_DAY3 指向「Day2 的輸出」，
#   例如：INPUT_CSV_DAY3 = OUTPUT_CSV_DAY2
from project_config import INPUT_CSV_DAY3 as INPUT_CSV
from project_config import OUTPUT_CSV_DAY3 as OUTPUT_CSV
from project_config import YES_RATIO_PLOT, TOPK

# ★ 目標欄名（僅用來「跳過處理」；Day4 會負責最終正規化/驗證）
try:
    from project_config import TARGET_COL as TARGET
except Exception:
    TARGET = "LUNG_CANCER"

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
    ★ 修正版：判斷是否為 YES/NO 欄位（更嚴格）
    - 若欄位名稱是目標欄（TARGET），直接回 False（Day3 不碰目標欄）
    - 數值型：唯一值非空且 ⊆ {0,1} 才算 YES/NO 題（全空或單一值不算）
    - 文字型：唯一值非空，且 ⊆ 允許集合 {"YES","NO","1","0","1.0","0.0","Y","N","是","否","TRUE","FALSE"}
    """
    # ★ Day3 不處理目標欄，避免把 label 誤轉成 NaN
    if s.name == TARGET:
        return False

    if pd.api.types.is_numeric_dtype(s):  # 數值欄位處理
        uniq_num = pd.Series(s.dropna().unique())
        if len(uniq_num) == 0:
            return False  # ★ 全空不算 YES/NO
        uniq_num = set(pd.to_numeric(uniq_num, errors="coerce"))
        return len(uniq_num) > 0 and uniq_num.issubset({0.0, 1.0})

    allowed = {"YES", "NO", "Y", "N", "1", "0", "1.0", "0.0", "TRUE", "FALSE", "是", "否"}
    vals = pd.Series(s.dropna().astype(str).str.strip().str.upper().unique())
    if len(vals) == 0:
        return False  # ★ 全空不算
    unexpected = [v for v in vals if v not in allowed]
    return len(unexpected) == 0

def coerce_yesno_to_binary(s: pd.Series) -> pd.Series:
    """
    將 YES/NO 等對映為 1/0；無法辨識者設為 NaN；數值型欄位直接通過
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")   # 如果本來就是數字（0/1），轉成數值型以避免物件型

    mapping = {
        "YES": 1, "Y": 1, "TRUE": 1, "是": 1, "1": 1, "1.0": 1,
        "NO": 0,  "N": 0, "FALSE": 0, "否": 0, "0": 0, "0.0": 0,
    }
    s_norm = normalize_yesno_series(s)
    return s_norm.map(mapping)

def find_smoking_columns(df: pd.DataFrame) -> list:
    """
    以關鍵字自動偵測吸菸相關欄位
    """
    keywords = [
        r"smok", r"cig", r"tobacco", r"pack", r"nicotin", r"pipe", r"quit",
        r"secondhand", r"passive", r"vape", r"e-?cig", r"hookah"
    ]
    pattern = re.compile("|".join(keywords), flags=re.IGNORECASE)  # 把清單裡的字樣用 | 分隔，忽略大小寫
    smoke_cols = [c for c in df.columns if pattern.search(str(c))]
    return smoke_cols   # 把找到的「吸菸相關欄位名稱清單」回傳

def add_bmi_and_obesity(df: pd.DataFrame) -> pd.DataFrame:
    """
    若存在 Height(公分) 與 Weight(公斤) 欄位，計算 BMI 與 Obesity(>=27)。
    欄名不分大小寫；會嘗試常見變體。
    """
    h_candidates = ["height", "身高", "Height", "HEIGHT"]
    w_candidates = ["weight", "體重", "Weight", "WEIGHT"]

    def find_col(df_local: pd.DataFrame, cands: list) -> str | None:
        for c in cands:
            if c in df_local.columns:
                return c
        # 寬鬆匹配：忽略大小寫
        lower_map = {col.lower(): col for col in df_local.columns}
        for c in cands:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    h_col = find_col(df, h_candidates)
    w_col = find_col(df, w_candidates)

    if h_col and w_col:
        h_cm = pd.to_numeric(df[h_col], errors="coerce")
        w_kg = pd.to_numeric(df[w_col], errors="coerce")
        h_m = h_cm / 100.0
        bmi = w_kg / (h_m ** 2)
        df["BMI"] = bmi
        df["Obesity"] = df["BMI"].apply(
            lambda x: "肥胖" if pd.notna(x) and x >= 27 else ("正常" if pd.notna(x) else pd.NA)
        )

    return df  
    # ← 一定要在函式內層（和 if 對齊），不要縮到 if 裡，也不要在函式外
    # 在 DataFrame 新增一個 "Obesity" 欄位，標記肥胖狀態
    # .apply(lambda x: ...)：對 "BMI" 欄位的每個值做判斷
    # 條件解釋：用pd.notna(x)語法
    #   - 如果 x 不是 NaN 且 >= 27 → 標記為 "肥胖" 套用.apply(lambda x: ...)規則
    #   - 如果 x 不是 NaN 且 < 27 → 標記為 "正常"  套用.apply(lambda x: ...)規則
    #   - 如果 x 是 NaN → 標記為缺失值 pd.NA    pd.notna(x)
    # 回傳更新後的 DataFrame，包含新計算的 BMI 與 Obesity 欄位

def run_day3(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not Path(input_csv).exists():
        print(f"[錯誤] 找不到輸入檔：{input_csv}")
        sys.exit(1)
    df = pd.read_csv(input_csv)
#  main() 就是程式的「入口函式」(entry point)。
#  任何資料處理、分析、存檔流程，都會在這裡被串接起來。
#  確保輸入存在
#  讀進同一份資料 (df)
#  一致性：所有處理 (清理、加 BMI、算 YES 比例、繪圖) 都針對同一份 df。
#  可控性：清楚知道資料從哪裡來、要存去哪裡。
#  可追蹤性：輸入 → 處理 → 輸出，流程清楚明瞭。
    
    # 1) 自動辨識並轉換 YES/NO 欄位為 1/0（★ 不包含目標欄）
    yesno_cols = []
    for col in df.columns:
        if is_yesno_column(df[col]):
            yesno_cols.append(col)

    # 2) 轉換（★ 再次保險：跳過目標欄）
    df_bin = df.copy()
    for col in yesno_cols:
        if col == TARGET:
            continue  # ★ Day3 不處理標籤欄
        df_bin[col] = coerce_yesno_to_binary(df_bin[col])
    """ 建立一個 df 的完整複本，命名為 df_bin。
    這樣做的好處是：不會直接修改原始資料 df 避免資料被覆蓋。 """

    # 3) 吸菸分數 / 風險分群
    smoke_cols = find_smoking_columns(df_bin)
    usable_smoke_cols = []
    for c in smoke_cols:
        tmp = pd.to_numeric(df_bin[c], errors="coerce")
        if tmp.notna().sum() > 0:
            df_bin[c] = tmp
            usable_smoke_cols.append(c)
    print(f"[偵測到的吸菸欄位] {smoke_cols}")
    print(f"[可用的吸菸欄位] {usable_smoke_cols}")
    if usable_smoke_cols:
        df_bin["smoke_score"] = df_bin[usable_smoke_cols].sum(axis=1, skipna=True)
        print(f"[smoke_score 使用欄位] {usable_smoke_cols}")
        df_bin["smoke_risk"] = df_bin["smoke_score"].apply(
            lambda s: "高風險" if pd.notna(s) and s >= 2 else ("低風險" if pd.notna(s) else pd.NA)
        )
    else:
        df_bin["smoke_score"] = pd.NA
        df_bin["smoke_risk"] = pd.NA
        print("[提醒] 沒有可用的吸菸欄位，smoke_score = NA")

    # 4) 在現有的 df_bin 基礎上，加 BMI/Obesity 欄位
    df_bin = add_bmi_and_obesity(df_bin)
    assert isinstance(df_bin, pd.DataFrame)

    # 5) YES 比例（每題平均，因 1=YES, 0=NO）→ 僅針對成功轉成 0/1 的欄位
    yes_ratio = pd.Series(dtype=float)
    print(f"[偵測到的 YES/NO 欄位] {yesno_cols}")
    if yesno_cols:
        # ★ 只對實際轉換成功且為數值的欄位計算
        numeric_yes_cols = [c for c in yesno_cols if pd.api.types.is_numeric_dtype(df_bin[c])]
        if numeric_yes_cols:
            yes_ratio = df_bin[numeric_yes_cols].mean(numeric_only=True).sort_values(ascending=False)
            print(f"[YES/NO 比例計算欄位] {numeric_yes_cols}")
            print(f"[YES/NO 比例結果] \n{yes_ratio}")
        else:
            print("[提醒] 沒有實際轉成數值的 YES/NO 欄位，略過 YES 比例統計。")
    else:
        print("[提醒] 沒有偵測到 YES/NO 欄位，略過 YES 比例統計。")

    # 6) 繪圖：YES 比例 TopK（有資料才畫）
    if yes_ratio.empty:
        print("[提醒] 無可計算的 YES/NO 欄位，略過繪圖。")
    else:
        try:
            topk = yes_ratio.head(min(TOPK, len(yes_ratio)))
            print(f"[準備繪圖的欄位 Top{len(topk)}] {topk.index.tolist()}")
            plt.figure(figsize=(max(8, len(topk) * 0.45), 4.8))
            topk.plot(kind="bar")
            plt.title(f"Top {len(topk)} YES 比例題目")
            plt.ylabel("YES 比例")
            plt.tight_layout()
            plt.savefig(YES_RATIO_PLOT, dpi=150)
            print(f"[完成] 已輸出柱狀圖：{YES_RATIO_PLOT}")
            plt.show()
            plt.close()
        except Exception as e:
            print(f"[警告] 繪圖失敗：{e}")

    # 7) 輸出強化後資料（無論是否繪圖都會執行）
    df_bin.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[完成] 已輸出 Day3 強化後資料：{output_csv}")

    # 8) 終端簡報（總結）
    print("\n=== Day3 摘要 ===")
    print(f"- 偵測並轉換 YES/NO 欄位數：{len(yesno_cols)}（已排除目標欄 {TARGET}）")
    print(f"- 偵測吸菸相關欄位（原始偵測）：{smoke_cols if len(smoke_cols) > 0 else '（無）'}")
    print(f"- 可用的吸菸欄位（成功轉數字）：{usable_smoke_cols if len(usable_smoke_cols) > 0 else '（無）'}")
    if "BMI" in df_bin.columns:
        valid_bmi = df_bin["BMI"].notna().sum()
        print(f"- 已新增 BMI 與 Obesity（有效 BMI 筆數：{valid_bmi}）")
    if not yes_ratio.empty:
        print(f"- YES 比例最高的 5 題：")
        print(yes_ratio.head(5).round(3))


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        inpath = Path(sys.argv[1])
        outpath = Path(sys.argv[2]) if len(sys.argv) >= 3 else OUTPUT_CSV
        run_day3(inpath, outpath)
    else:
        run_day3()
