# Полный пересчёт экспериментов под Windows (PowerShell 5.1 и 7+).
#
# Запуск из папки «Код_и_результаты»:
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1              # эксперименты, рисунки, проверки
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Force       # пересчитать всё заново
#
# Уже посчитанные сценарии пропускаются (как в run_all.sh): полный пересчёт — около 3 ч
# на двух ядрах. Готовые таблицы results\ уже лежат в папке, поэтому без -Force скрипт
# сразу перейдёт к рисункам и проверкам.

param(
    [switch]$Force,      # удалить готовые CSV и посчитать все эксперименты заново
    [switch]$Install,    # установить недостающие пакеты
    [string]$Index = "", # другой адрес индекса пакетов (зеркало PyPI), если pypi.org недоступен
    [string]$Python = "" # явный путь к python.exe (по умолчанию — активное окружение conda или python из PATH)
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"            # UTF-8 для print и файлов (символы ξ, σ, λ в выводе)
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "1"       # параллелизм — через joblib, без вложенных потоков BLAS

$Root = $PSScriptRoot
$Src = Join-Path $Root "src"
$Res = Join-Path $Root "results"

# ---------------------------------------------------------------- интерпретатор Python
function Find-Python {
    # Порядок: явно заданный путь → активное окружение conda → python из PATH → лаунчер py.
    # Окружение conda идёт раньше лаунчера py: иначе при установленном также python.org
    # скрипт взял бы интерпретатор без нужных пакетов.
    $cands = @()
    if ($Python) { $cands += , @($Python) }
    if ($env:CONDA_PREFIX) {
        foreach ($p in @((Join-Path $env:CONDA_PREFIX "python.exe"), (Join-Path $env:CONDA_PREFIX "bin/python"))) {
            if (Test-Path $p) { $cands += , @($p) }
        }
    }
    $cands += , @("python")
    $cands += , @("py", "-3")
    $cands += , @("python3")
    foreach ($cand in $cands) {
        $exe = $cand[0]
        if ((Test-Path $exe) -or (Get-Command $exe -ErrorAction SilentlyContinue)) {
            $pre = @($cand | Select-Object -Skip 1)
            try {
                $v = & $exe @pre -c "import sys; print(sys.version_info >= (3, 9))" 2>$null
                if ($v -eq "True") { $r = @($exe) + $pre; return , $r }
            } catch { }
        }
    }
    throw "Не найден Python 3.9+. Установите Miniforge или Python с python.org (см. README)."
}
$Py = Find-Python
$PyExe = $Py[0]
$PyPre = @($Py | Select-Object -Skip 1)

function Invoke-Py {
    param([string[]]$PyArgs)
    & $PyExe @PyPre @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "Ошибка при выполнении: python $($PyArgs -join ' ')" }
}

Write-Host "Python: $(& $PyExe @PyPre -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# ---------------------------------------------------------------- зависимости
# модуль Python -> имя пакета для pip
$Need = [ordered]@{ numpy = "numpy"; pandas = "pandas"; scipy = "scipy"; sklearn = "scikit-learn";
                    matplotlib = "matplotlib>=3.9"; pyod = "pyod"; joblib = "joblib"; PIL = "pillow";
                    nbformat = "nbformat" }

function Get-Missing {
    $mods = ($Need.Keys | ForEach-Object { "'$_'" }) -join ", "
    $code = "import importlib.util as u; print(' '.join(m for m in [$mods] if u.find_spec(m) is None))"
    $out = (& $PyExe @PyPre -c $code)
    if ($out) { return @($out.Trim().Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)) }
    return @()
}

$missing = Get-Missing
if ($missing.Count -gt 0 -and $Install) {
    $pkgs = @($missing | ForEach-Object { $Need[$_] })
    Write-Host "установка: $($pkgs -join ', ')"
    # увеличенные таймаут и число повторов: при медленном соединении с pypi.org
    # стандартных 15 с не хватает (ошибка ReadTimeoutError)
    $pipArgs = @("-m", "pip", "install", "--timeout", "120", "--retries", "10", "--prefer-binary")
    if ($Index) { $pipArgs += @("--index-url", $Index) }
    Invoke-Py ($pipArgs + $pkgs)
    $missing = Get-Missing
}
if ($missing.Count -gt 0) {
    $pkgs = @($missing | ForEach-Object { $Need[$_] })
    throw ("Не установлены пакеты: $($pkgs -join ', '). Запустите скрипт с ключом -Install " +
           "(при проблемах с доступом к pypi.org — ещё и с -Index <адрес зеркала>).")
}

# ---------------------------------------------------------------- эксперименты
$Jobs = @(
    @("exp3g_optimizer.csv",      "exp3g_optimizer.py"),
    @("exp1_magnitude.csv",       "exp1_magnitude.py"),
    @("exp1b_edge.csv",           "exp1b_edge.py"),
    @("exp3a_x_outliers.csv",     "exp3_extensions.py", "3a"),
    @("exp3c_n_scan.csv",         "exp3_extensions.py", "3c"),
    @("exp3d_sample_size.csv",    "exp3_extensions.py", "3d"),
    @("exp2_peeling.csv",         "exp2_peeling.py"),
    @("exp2c_clean.csv",          "exp2c_clean.py"),
    @("exp3e_downstream.csv",     "exp3_extensions.py", "3e"),
    @("exp3b_xi_sensitivity.csv", "exp3_extensions.py", "3b"),
    @("exp4_sample_size.csv",     "exp4_sample_size.py")
)

Push-Location $Src
try {
    foreach ($job in $Jobs) {
        $csv = Join-Path $Res $job[0]
        $scriptArgs = @($job | Select-Object -Skip 1)
        if ($Force -and (Test-Path $csv)) { Remove-Item $csv }
        if ($job[0] -eq "exp2_peeling.csv" -and $Force) {
            $traj = Join-Path $Res "exp2_trajectories.csv"
            if (Test-Path $traj) { Remove-Item $traj }
        }
        if (Test-Path $csv) {
            Write-Host "пропуск: $($job[0]) уже посчитан"
        } else {
            Write-Host "расчёт: $($scriptArgs -join ' ')"
            $t0 = Get-Date
            Invoke-Py $scriptArgs
            Write-Host ("  готово за {0:N1} мин" -f ((Get-Date) - $t0).TotalMinutes)
        }
    }
    Invoke-Py @("paper_figures.py")
    Invoke-Py @("notebook_figures.py")
    Invoke-Py @("verify.py")
} finally {
    Pop-Location
}

Write-Host "ВСЁ ГОТОВО"
