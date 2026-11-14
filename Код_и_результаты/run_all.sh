#!/bin/bash
# Полный пересчёт всех экспериментов (около 3 ч на двух ядрах).
# Уже посчитанные сценарии пропускаются — удалите CSV, чтобы пересчитать.
cd "$(dirname "$0")/src"
export OMP_NUM_THREADS=1
R=../results
run() { [ -f "$R/$1" ] || python3 "${@:2}"; }
run exp3g_optimizer.csv       exp3g_optimizer.py
run exp1_magnitude.csv        exp1_magnitude.py
run exp1b_edge.csv            exp1b_edge.py
run exp3a_x_outliers.csv      exp3_extensions.py 3a
run exp3c_n_scan.csv          exp3_extensions.py 3c
run exp3d_sample_size.csv     exp3_extensions.py 3d
run exp2_peeling.csv          exp2_peeling.py
run exp2c_clean.csv           exp2c_clean.py
run exp3e_downstream.csv      exp3_extensions.py 3e
run exp3b_xi_sensitivity.csv  exp3_extensions.py 3b
run exp4_sample_size.csv      exp4_sample_size.py
python3 paper_figures.py
python3 notebook_figures.py
python3 verify.py
echo "ВСЁ ГОТОВО"
