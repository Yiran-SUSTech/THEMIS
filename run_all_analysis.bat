@echo off
chcp 65001 >nul
cd /d d:\THEMIS

echo ============================================================
echo  批量运行 extract_and_analyze_scores.py
echo ============================================================
echo.

echo [1/17] Sys_DiT_noexpert Human_DiT_500
python extract_and_analyze_scores.py --source Sys_DiT_noexpert Human_DiT_500 --vendi-ratio-csv "d:\THEMIS\vendi-DiT-5000\per_class_vendi_scores.csv"
echo.

echo [2/17] Sys_DiT_noref Human_DiT_500
python extract_and_analyze_scores.py --source Sys_DiT_noref Human_DiT_500 --vendi-ratio-csv "d:\THEMIS\vendi-DiT-5000\per_class_vendi_scores.csv"
echo.

echo [3/17] Sys_DiT_ref_500 Human_DiT_500
python extract_and_analyze_scores.py --source Sys_DiT_ref_500 Human_DiT_500 --vendi-ratio-csv "d:\THEMIS\vendi-DiT-5000\per_class_vendi_scores.csv"
echo.

echo [4/17] Sys_DiT_ref_cls_1k Human_DiT_500
python extract_and_analyze_scores.py --source Sys_DiT_ref_cls_1k Human_DiT_500
echo.

echo [5/17] Sys_IMF_ref_500 Human_IMF_500
python extract_and_analyze_scores.py --source Sys_IMF_ref_500 Human_IMF_500 --vendi-ratio-csv "d:\THEMIS\vendi-IMF-XL-2-5000\per_class_vendi_scores.csv"
echo.

echo [6/17] Sys_JiTfdloss_ref_500 Human_JiTfdloss_500
python extract_and_analyze_scores.py --source Sys_JiTfdloss_ref_500 Human_JiTfdloss_500 --vendi-ratio-csv "d:\THEMIS\vendi-JiT-FDloss-5000\per_class_vendi_scores.csv"
echo.

echo [7/17] Sys_IMFfdloss_ref_500 Human_IMFfdloss_500
python extract_and_analyze_scores.py --source Sys_IMFfdloss_ref_500 Human_IMFfdloss_500 --vendi-ratio-csv "d:\THEMIS\vendi-IMF-FDloss-5000\per_class_vendi_scores.csv"
echo.

echo [8/17] Sys_RAEv2_ref_500 Human_JiTfdloss_500
python extract_and_analyze_scores.py --source Sys_RAEv2_ref_500 Human_JiTfdloss_500 --vendi-ratio-csv "d:\THEMIS\vendi-RAEv2-5000\per_class_vendi_scores.csv"
echo.

echo [9/17] Sys_Val_ref_500 Human_Val_500
python extract_and_analyze_scores.py --source Sys_Val_ref_500 Human_Val_500
echo.

echo [10/17] Sys_VAR_noexpert Human_VAR_500
python extract_and_analyze_scores.py --source Sys_VAR_noexpert Human_VAR_500 --vendi-ratio-csv "d:\THEMIS\vendi-VAR-5000\per_class_vendi_scores.csv"
echo.

echo [11/17] Sys_VAR_noref Human_VAR_500
python extract_and_analyze_scores.py --source Sys_VAR_noref Human_VAR_500 --vendi-ratio-csv "d:\THEMIS\vendi-VAR-5000\per_class_vendi_scores.csv"
echo.

echo [12/17] Sys_VAR_ref_200 Human_VAR_200
python extract_and_analyze_scores.py --source Sys_VAR_ref_200 Human_VAR_200
echo.

echo [13/17] Sys_VAR_ref_400 Human_VAR_400
python extract_and_analyze_scores.py --source Sys_VAR_ref_400 Human_VAR_400
echo.

echo [14/17] Sys_VAR_ref_500 Human_VAR_500
python extract_and_analyze_scores.py --source Sys_VAR_ref_500 Human_VAR_500
echo.

echo [15/17] Sys_VAR_ref_600 Human_VAR_600
python extract_and_analyze_scores.py --source Sys_VAR_ref_600 Human_VAR_600
echo.

echo [16/17] Sys_VAR_ref_800 Human_VAR_800
python extract_and_analyze_scores.py --source Sys_VAR_ref_800 Human_VAR_800
echo.

echo [17/17] Sys_VAR_ref_1000 Human_VAR_1000
python extract_and_analyze_scores.py --source Sys_VAR_ref_1000 Human_VAR_1000
echo.

echo ============================================================
echo  全部 17 个任务已完成!
echo ============================================================
pause
