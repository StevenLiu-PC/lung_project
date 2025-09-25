# Day2 pipeline: 清理與初步特徵工程
# 註解：僅新增說明，不影響程式邏輯

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # ? lung_project ?寧??
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from project_config import PROC_OUT, OUTPUT_DAY2

def run_day2():
    # 霈??Day1 頛詨嚗?嗆雿??砌??挾頝???df_processed嚗?
    if not Path(PROC_OUT).exists():
        raise FileNotFoundError(f"[?航炊] ?曆???Day1 頛詨嚗PROC_OUT}嚗??銵?Day1??)
    df_processed = pd.read_csv(PROC_OUT, encoding="utf-8-sig")

    cat_cols = df_processed.select_dtypes(include="object").columns.tolist()
    yesno_cols = []  # ?園??迤??YES/NO 甈??迂

    for col in cat_cols:
        # 撠迨甈?璅???頧?銝脯之撖怒??蝛箇嚗??冽瑼Ｘ?澆?
        vals = df_processed[col].astype(str).str.upper().str.strip()
        uniq = set(vals.dropna().unique())  # ?駁蝻箏仃?澆????臭??潮???
        if uniq.issubset({"YES", "NO"}):   # ?芣? YES/NO ???箔???雿?
            yesno_cols.append(col)

    print(f"[Day2] ?菜葫??YES/NO 甈?嚗yesno_cols if yesno_cols else '嚗嚗?}")

    # 撠儘霅??YES/NO 甈???撠?頧 float ?嫣噶敺?蝯梯??撩憭勗潸???
    for col in yesno_cols:
        df_processed[col] = (
            df_processed[col]
            .astype(str).str.upper().str.strip()   # ?炎?交?銝?渡?甇????
            .map({"YES": 1, "NO": 0})              # YES??, NO??
            .astype("float")                        # 頧?潘??嫣噶 mean() 蝑?雿?
        )

    # Step 11嚗炎?亦撩憭勗潭?瘜?
    # 憿舐內??雿?NaN ???敹恍??⊿?閬???甈?
    print("\n=== 蝻箏仃?潛絞閮???雿?NaN 蝑嚗?==")
    na_counts = df_processed.isna().sum()
    print(na_counts[na_counts > 0].sort_values(ascending=False) if na_counts.any() else "?∠撩憭勗?)

    # Step 12嚗撩憭勗潸???蝷箇?嚗?
    # ??嚗?
    # - ?詨潭?雿?int/float嚗? 隞亙像?澆‵鋆?
    # - 隞摮葡??雿?object嚗? 隞亦?詨‵鋆?
    # 瘜冽?嚗迤撘?獢靘?憓?箔葉雿?摰潭??脤?憛怨?瘜?
   
    # 12.1 ?詨潭?雿???瘨菔? int ??float嚗?
    num_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_processed[col].isna().any():
            mean_val = df_processed[col].mean()
            df_processed[col] = df_processed[col].fillna(mean_val)

    # 12.2 隞摮葡??雿??亙??剁?
    obj_cols = df_processed.select_dtypes(include="object").columns
    for col in obj_cols:
        if df_processed[col].isna().any():
            mode_val = df_processed[col].mode(dropna=True)
            if not mode_val.empty:
                df_processed[col] = df_processed[col].fillna(mode_val.iloc[0])

    print("\n[Day2] 蝻箏仃?潸?????)

    
    # Step 13嚗???雿炎閬?
    # - ?亙???AGE嚗撓?箏?膩蝯梯?
    # - 撠?LUNG_CANCER 憿舐內閮?????嚗噶?潭炎?仿??乩?撟唾﹛
   
    if "AGE" in df_processed.columns:
        print("\n=== 撟湧翩??嚗GE.describe嚗?==")
        print(df_processed["AGE"].describe())

    print("\n=== LUNG_CANCER ??嚗???/ ?曉?瘥?===")
    if "LUNG_CANCER" in df_processed.columns:
        counts = df_processed["LUNG_CANCER"].value_counts(dropna=False)
        perc = (df_processed["LUNG_CANCER"].value_counts(normalize=True, dropna=False) * 100).round(2)
        print("閮嚗?)
        print(counts)
        print("\n?曉?瘥?%嚗?")
        print(perc)
    else:
        print("?曆???LUNG_CANCER 甈?")

    
    # Step 14嚗撓??Day 2 ??敺?瑼?
    # ??Day 1 ??摮嚗????雿輻 UTF-8 蝣箔?銝剜?銝?蝣?
    
    OUTPUT_DAY2.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(OUTPUT_DAY2, index=False, encoding="utf-8-sig")
    print(f"??Day 2 CSV 撌脣摮?{OUTPUT_DAY2}")


    # Step 15 ??????
    if "AGE" in df_processed.columns:

        # ??嚗?憒? 5 甇脖?蝯?
        bin_size = 5
        age_bins = pd.cut(df_processed["AGE"],
                          bins=range(int(df_processed["AGE"].min()), int(df_processed["AGE"].max()) + bin_size, bin_size))

        # 閮?瘥?蝑
        age_group_counts = age_bins.value_counts().sort_index()

        # ?其葉雿?嗡? x 頠詨漣璅?
        bin_midpoints = [interval.mid for interval in age_group_counts.index]

        plt.plot(bin_midpoints, age_group_counts.values, marker='o', linewidth=2, color="teal")
        plt.title("Age Distribution (Grouped Count)")
        plt.xlabel("Age Group Midpoint")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
       
        plots_dir = Path("data_lung/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / "day2_age_grouped_line.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[Day2] 撌脰撓?箏?瑼?{out}")
if __name__ == "__main__":
    run_day2()

