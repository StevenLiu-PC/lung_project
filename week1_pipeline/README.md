# week1_pipeline
Week1 的每日流程腳本（可獨立執行）。
- `day1_pipeline.py` ~ `day7_pipeline.py`：資料讀取、清理、特徵工程、切分、可視化
- `day8_pipeline.py`：資料品質檢查（QC）與清理
- `day9_pipeline.py`：指標彙整與文字報告
- `day10_pipeline.py`：多模型交叉驗證（LR/DT/RF/SVM），依 `roc_auc` 選最優並重訓

## 執行
```bash
python .\week1_pipeline\day8_pipeline.py
python .\week1_pipeline\day9_pipeline.py
python .\week1_pipeline\day10_pipeline.py
```
