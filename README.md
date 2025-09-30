# Lung Project — Week 1 Pipelines

## Day 1–7（基線）
- Day 1–5：專案結構、嚴格目標欄標準化（Day4）、基礎特徵工程（Day5）
- Day 6：交叉驗證骨架、指標與報表
- Day 7：多模型比較、ROC/係數/特徵重要度

## Day 8–10（本 PR 新增）
- **Day 8**：資料品質檢查（QC）與清理
- **Day 9**：指標彙整與文字報告
- **Day 10**：多模型交叉驗證（LR/DT/RF/SVM），依 `roc_auc` 選最優並重訓  
  產物：`data_lung/artifacts/day10/day10_report.txt`, `cv_results_day10.csv`

## 執行
```bash
python .\week1_pipeline\day8_pipeline.py
python .\week1_pipeline\day9_pipeline.py
python .\week1_pipeline\day10_pipeline.py
