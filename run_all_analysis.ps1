Set-Location d:\THEMIS

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Batch running extract_and_analyze_scores.py" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$commands = @(
    @{ desc = "Sys_DiT_noexpert Human_DiT_500";         args = @("--source", "Sys_DiT_noexpert", "Human_DiT_500", "--vendi-ratio-csv", "d:\THEMIS\vendi-DiT-5000\per_class_vendi_scores.csv") },
    @{ desc = "Sys_DiT_noref Human_DiT_500";             args = @("--source", "Sys_DiT_noref", "Human_DiT_500", "--vendi-ratio-csv", "d:\THEMIS\vendi-DiT-5000\per_class_vendi_scores.csv") },
    @{ desc = "Sys_DiT_ref_500 Human_DiT_500";           args = @("--source", "Sys_DiT_ref_500", "Human_DiT_500", "--vendi-ratio-csv", "d:\THEMIS\vendi-DiT-5000\per_class_vendi_scores.csv") },
    # @{ desc = "Sys_DiT_ref_cls_1k Human_DiT_500";        args = @("--source", "Sys_DiT_ref_cls_1k", "Human_DiT_500") },
    @{ desc = "Sys_IMF_ref_500 Human_IMF_500";           args = @("--source", "Sys_IMF_ref_500", "Human_IMF_500", "--vendi-ratio-csv", "d:\THEMIS\vendi-IMF-XL-2-5000\per_class_vendi_scores.csv") },
    @{ desc = "Sys_JiTfdloss_ref_500 Human_JiTfdloss_500"; args = @("--source", "Sys_JiTfdloss_ref_500", "Human_JiTfdloss_500", "--vendi-ratio-csv", "d:\THEMIS\vendi-JiT-FDloss-5000\per_class_vendi_scores.csv") },
    @{ desc = "Sys_IMFfdloss_ref_500 Human_IMFfdloss_500"; args = @("--source", "Sys_IMFfdloss_ref_500", "Human_IMFfdloss_500", "--vendi-ratio-csv", "d:\THEMIS\vendi-IMF-FDloss-5000\per_class_vendi_scores.csv") },
    @{ desc = "Sys_RAEv2_ref_500 Human_JiTfdloss_500";   args = @("--source", "Sys_RAEv2_ref_500", "Human_JiTfdloss_500", "--vendi-ratio-csv", "d:\THEMIS\vendi-RAEv2-5000\per_class_vendi_scores.csv") },
    @{ desc = "Sys_Val_ref_500 Human_Val_500";           args = @("--source", "Sys_Val_ref_500", "Human_Val_500") },
    @{ desc = "Sys_VAR_noexpert Human_VAR_500";          args = @("--source", "Sys_VAR_noexpert", "Human_VAR_500", "--vendi-ratio-csv", "d:\THEMIS\vendi-VAR-5000\per_class_vendi_scores.csv") },
    @{ desc = "Sys_VAR_noref Human_VAR_500";             args = @("--source", "Sys_VAR_noref", "Human_VAR_500", "--vendi-ratio-csv", "d:\THEMIS\vendi-VAR-5000\per_class_vendi_scores.csv") },
    @{ desc = "Sys_VAR_ref_200 Human_VAR_200";           args = @("--source", "Sys_VAR_ref_200", "Human_VAR_200") },
    @{ desc = "Sys_VAR_ref_400 Human_VAR_400";           args = @("--source", "Sys_VAR_ref_400", "Human_VAR_400") },
    @{ desc = "Sys_VAR_ref_500 Human_VAR_500";           args = @("--source", "Sys_VAR_ref_500", "Human_VAR_500", "--vendi-ratio-csv", "d:\THEMIS\vendi-VAR-5000\per_class_vendi_scores.csv") },
    @{ desc = "Sys_VAR_ref_600 Human_VAR_600";           args = @("--source", "Sys_VAR_ref_600", "Human_VAR_600") },
    @{ desc = "Sys_VAR_ref_800 Human_VAR_800";           args = @("--source", "Sys_VAR_ref_800", "Human_VAR_800") },
    @{ desc = "Sys_VAR_ref_1000 Human_VAR_1000";         args = @("--source", "Sys_VAR_ref_1000", "Human_VAR_1000") }
)

$total = $commands.Count
for ($i = 0; $i -lt $total; $i++) {
    $n = $i + 1
    $cmd = $commands[$i]
    Write-Host "[$n/$total] $($cmd.desc)" -ForegroundColor Yellow
    Write-Host ""
    & python extract_and_analyze_scores.py @($cmd.args)
    Write-Host ""
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host "  All $total tasks completed!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
