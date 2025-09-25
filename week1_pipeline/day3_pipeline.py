# Day3 pipeline: 切分/可視化/檢查
# 註解：僅新增說明，不影響程式邏輯

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # lung_project 鞈?憭?
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]  # 甇??擃??踹?銝剜?摮?????
plt.rcParams["axes.unicode_minus"] = False  # ?踹?鞎?霈?蝣?

# ??撱箄降??project_config.py ?改?霈?INPUT_CSV_DAY3 ???ay2 ?撓?箝?
#   靘?嚗NPUT_CSV_DAY3 = OUTPUT_CSV_DAY2
from project_config import INPUT_CSV_DAY3 as INPUT_CSV
from project_config import OUTPUT_CSV_DAY3 as OUTPUT_CSV
from project_config import YES_RATIO_PLOT, TOPK

# ???格?甈?嚗??其??歲????Day4 ??鞎祆?蝯迤閬?/撽?嚗?
try:
    from project_config import TARGET_COL as TARGET
except Exception:
    TARGET = "LUNG_CANCER"

# Day3 ??啁?頝臬?皞?
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
YES_RATIO_PLOT.parent.mkdir(parents=True, exist_ok=True)

def normalize_yesno_series(s: pd.Series) -> pd.Series:
    """皜?鞈?"""
    if s.dtype == object:
        return s.astype(str).str.strip().str.upper()
    return s
                                    # .astype(str) ???潸???銝莎??踹?瘛瑕?詨??粹
                                    # .str.strip() ???餅???蝛箇
                                    # .str.upper() ???券頧之撖恬??踹?憭批?撖思?銝??
                                 
def is_yesno_column(s: pd.Series) -> bool:
    """
    ??靽格迤???斗?臬??YES/NO 甈?嚗?湔嚗?
    - ?交?雿?蝔望?格?甈?TARGET嚗??湔??False嚗ay3 銝１?格?甈?
    - ?詨澆?嚗銝?潮?蝛箔? ??{0,1} ?? YES/NO 憿??函征?銝?潔?蝞?
    - ?????臭??潮?蝛綽?銝????迂?? {"YES","NO","1","0","1.0","0.0","Y","N","??,"??,"TRUE","FALSE"}
    """
    # ??Day3 銝??璅?嚗?? label 隤方???NaN
    if s.name == TARGET:
        return False

    if pd.api.types.is_numeric_dtype(s):  # ?詨潭?雿???
        uniq_num = pd.Series(s.dropna().unique())
        if len(uniq_num) == 0:
            return False  # ???函征銝? YES/NO
        uniq_num = set(pd.to_numeric(uniq_num, errors="coerce"))
        return len(uniq_num) > 0 and uniq_num.issubset({0.0, 1.0})

    allowed = {"YES", "NO", "Y", "N", "1", "0", "1.0", "0.0", "TRUE", "FALSE", "??, "??}
    vals = pd.Series(s.dropna().astype(str).str.strip().str.upper().unique())
    if len(vals) == 0:
        return False  # ???函征銝?
    unexpected = [v for v in vals if v not in allowed]
    return len(unexpected) == 0

def coerce_yesno_to_binary(s: pd.Series) -> pd.Series:
    """
    撠?YES/NO 蝑?? 1/0嚗瘜儘霅身??NaN嚗?澆?甈??湔??
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")   # 憒??砌?撠望?詨?嚗?/1嚗?頧??詨澆?隞仿?隞嗅?

    mapping = {
        "YES": 1, "Y": 1, "TRUE": 1, "??: 1, "1": 1, "1.0": 1,
        "NO": 0,  "N": 0, "FALSE": 0, "??: 0, "0": 0, "0.0": 0,
    }
    s_norm = normalize_yesno_series(s)
    return s_norm.map(mapping)

def find_smoking_columns(df: pd.DataFrame) -> list:
    """
    隞仿??萄??芸??菜葫?貉?賊?甈?
    """
    keywords = [
        r"smok", r"cig", r"tobacco", r"pack", r"nicotin", r"pipe", r"quit",
        r"secondhand", r"passive", r"vape", r"e-?cig", r"hookah"
    ]
    pattern = re.compile("|".join(keywords), flags=re.IGNORECASE)  # ???株ㄐ??璅? | ??嚗蕭?亙之撠神
    smoke_cols = [c for c in df.columns if pattern.search(str(c))]
    return smoke_cols   # ??啁???貊??雿?蝔望??柴???

def add_bmi_and_obesity(df: pd.DataFrame) -> pd.DataFrame:
    """
    ?亙???Height(?砍?) ??Weight(?祆) 甈?嚗?蝞?BMI ??Obesity(>=27)??
    甈?銝?憭批?撖恬???閰血虜閬?擃?
    """
    h_candidates = ["height", "頨恍?", "Height", "HEIGHT"]
    w_candidates = ["weight", "擃?", "Weight", "WEIGHT"]

    def find_col(df_local: pd.DataFrame, cands: list) -> str | None:
        for c in cands:
            if c in df_local.columns:
                return c
        # 撖祇??寥?嚗蕭?亙之撠神
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
            lambda x: "?亥?" if pd.notna(x) and x >= 27 else ("甇?虜" if pd.notna(x) else pd.NA)
        )

    return df  
    # ??銝摰??典撘撅歹???if 撠?嚗?銝?蝮桀 if 鋆∴?銋?閬?賢?憭?
    # ??DataFrame ?啣?銝??"Obesity" 甈?嚗?閮????
    # .apply(lambda x: ...)嚗? "BMI" 甈????澆??斗
    # 璇辣閫??嚗pd.notna(x)隤?
    #   - 憒? x 銝 NaN 銝?>= 27 ??璅???"?亥?" 憟.apply(lambda x: ...)閬?
    #   - 憒? x 銝 NaN 銝?< 27 ??璅???"甇?虜"  憟.apply(lambda x: ...)閬?
    #   - 憒? x ??NaN ??璅??箇撩憭勗?pd.NA    pd.notna(x)
    # ??湔敺? DataFrame嚗??急閮???BMI ??Obesity 甈?

def run_day3(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    if not Path(input_csv).exists():
        print(f"[?航炊] ?曆??啗撓?交?嚗input_csv}")
        sys.exit(1)
    df = pd.read_csv(input_csv)
#  main() 撠望蝔????撘?entry point)??
#  隞颱?鞈???????瑼?蝔??賣??券ㄐ鋡思葡?亥絲靘?
#  蝣箔?頛詨摮
#  霈?脣?銝隞質???(df)
#  銝?湔改??????(皜??? BMI?? YES 瘥??鼓?? ?賡?撠?銝隞?df??
#  ?舀?改?皜??仿?鞈?敺鋆∩???摮?芾ㄐ??
#  ?航蕭頩斗改?頛詨 ???? ??頛詨嚗?蝔?璆??准?
    
    # 1) ?芸?颲刻?銝西???YES/NO 甈???1/0嚗? 銝??怎璅?嚗?
    yesno_cols = []
    for col in df.columns:
        if is_yesno_column(df[col]):
            yesno_cols.append(col)

    # 2) 頧?嚗? ?活靽嚗歲?璅?嚗?
    df_bin = df.copy()
    for col in yesno_cols:
        if col == TARGET:
            continue  # ??Day3 銝???蝐斗?
        df_bin[col] = coerce_yesno_to_binary(df_bin[col])
    """ 撱箇?銝??df ???渲??穿??賢???df_bin??
    ?見??憟質??荔?銝??湔靽格??鞈? df ?踹?鞈?鋡怨???"""

    # 3) ?貉? / 憸券?黎
    smoke_cols = find_smoking_columns(df_bin)
    usable_smoke_cols = []
    for c in smoke_cols:
        tmp = pd.to_numeric(df_bin[c], errors="coerce")
        if tmp.notna().sum() > 0:
            df_bin[c] = tmp
            usable_smoke_cols.append(c)
    print(f"[?菜葫?啁??貉甈?] {smoke_cols}")
    print(f"[?舐??豢?雿 {usable_smoke_cols}")
    if usable_smoke_cols:
        df_bin["smoke_score"] = df_bin[usable_smoke_cols].sum(axis=1, skipna=True)
        print(f"[smoke_score 雿輻甈?] {usable_smoke_cols}")
        df_bin["smoke_risk"] = df_bin["smoke_score"].apply(
            lambda s: "擃◢?? if pd.notna(s) and s >= 2 else ("雿◢?? if pd.notna(s) else pd.NA)
        )
    else:
        df_bin["smoke_score"] = pd.NA
        df_bin["smoke_risk"] = pd.NA
        print("[??] 瘝??舐??豢?雿?smoke_score = NA")

    # 4) ?函?? df_bin ?箇?銝???BMI/Obesity 甈?
    df_bin = add_bmi_and_obesity(df_bin)
    assert isinstance(df_bin, pd.DataFrame)

    # 5) YES 瘥?嚗?憿像????1=YES, 0=NO嚗? ??撠?????0/1 ??雿?
    yes_ratio = pd.Series(dtype=float)
    print(f"[?菜葫?啁? YES/NO 甈?] {yesno_cols}")
    if yesno_cols:
        # ???芸?撖阡?頧???銝?詨潛?甈?閮?
        numeric_yes_cols = [c for c in yesno_cols if pd.api.types.is_numeric_dtype(df_bin[c])]
        if numeric_yes_cols:
            yes_ratio = df_bin[numeric_yes_cols].mean(numeric_only=True).sort_values(ascending=False)
            print(f"[YES/NO 瘥?閮?甈?] {numeric_yes_cols}")
            print(f"[YES/NO 瘥?蝯?] \n{yes_ratio}")
        else:
            print("[??] 瘝?撖阡?頧??詨潛? YES/NO 甈?嚗??YES 瘥?蝯梯???)
    else:
        print("[??] 瘝??菜葫??YES/NO 甈?嚗??YES 瘥?蝯梯???)

    # 6) 蝜芸?嚗ES 瘥? TopK嚗?鞈??嚗?
    if yes_ratio.empty:
        print("[??] ?∪閮???YES/NO 甈?嚗?鼓??)
    else:
        try:
            topk = yes_ratio.head(min(TOPK, len(yes_ratio)))
            print(f"[皞?蝜芸???雿?Top{len(topk)}] {topk.index.tolist()}")
            plt.figure(figsize=(max(8, len(topk) * 0.45), 4.8))
            topk.plot(kind="bar")
            plt.title(f"Top {len(topk)} YES 瘥?憿")
            plt.ylabel("YES 瘥?")
            plt.tight_layout()
            plt.savefig(YES_RATIO_PLOT, dpi=150)
            print(f"[摰?] 撌脰撓?箸???{YES_RATIO_PLOT}")
            plt.show()
            plt.close()
        except Exception as e:
            print(f"[霅血?] 蝜芸?憭望?嚗e}")

    # 7) 頛詨撘瑕?敺????∟??臬蝜芸??賣??瑁?嚗?
    df_bin.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[摰?] 撌脰撓??Day3 撘瑕?敺???{output_csv}")

    # 8) 蝯垢蝪∪嚗蜇蝯?
    print("\n=== Day3 ?? ===")
    print(f"- ?菜葫銝西???YES/NO 甈??賂?{len(yesno_cols)}嚗歇??格?甈?{TARGET}嚗?)
    print(f"- ?菜葫?貉?賊?甈?嚗?憪皜穿?嚗smoke_cols if len(smoke_cols) > 0 else '嚗嚗?}")
    print(f"- ?舐??豢?雿???頧摮?嚗usable_smoke_cols if len(usable_smoke_cols) > 0 else '嚗嚗?}")
    if "BMI" in df_bin.columns:
        valid_bmi = df_bin["BMI"].notna().sum()
        print(f"- 撌脫憓?BMI ??Obesity嚗???BMI 蝑嚗valid_bmi}嚗?)
    if not yes_ratio.empty:
        print(f"- YES 瘥??擃? 5 憿?")
        print(yes_ratio.head(5).round(3))


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        inpath = Path(sys.argv[1])
        outpath = Path(sys.argv[2]) if len(sys.argv) >= 3 else OUTPUT_CSV
        run_day3(inpath, outpath)
    else:
        run_day3()

