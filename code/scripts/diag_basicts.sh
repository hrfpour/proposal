#!/usr/bin/env bash
# Phase 3 diagnostic: prints how BasicTS 1.0 defines models, configs and the training taskflow.
# Run from the repo root (proposal/):   bash code/scripts/diag_basicts.sh > ~/Downloads/diag_basicts.txt 2>&1
BT=code/BasicTS

show() {   # show <relative path> <max lines>
  echo; echo "################ $1 ################"
  if [ -f "$BT/$1" ]; then head -n "${2:-200}" "$BT/$1"; else echo "(missing)"; fi
}
ls_dir() { # ls_dir <relative dir>
  echo; echo "################ ls $1 ################"
  ls -la "$BT/$1" 2>&1 | head -n 60
}
show_class() {  # show_class <class name> <max lines>: find the file that defines it
  f=$(grep -rl "class $1" "$BT/src" 2>/dev/null | head -n 1)
  echo; echo "################ class $1 -> ${f:-NOT FOUND} ################"
  if [ -n "$f" ]; then head -n "${2:-200}" "$f"; fi
}

echo "BasicTS commit: $(git -C "$BT" rev-parse --short HEAD 2>&1)"

ls_dir docs
ls_dir src/basicts
ls_dir src/basicts/configs
ls_dir src/basicts/models/STID
ls_dir src/basicts/models/StemGNN
ls_dir src/basicts/runners
ls_dir src/basicts/runners/taskflow
ls_dir src/basicts/metrics

show src/basicts/models/__init__.py 80
show src/basicts/models/STID/__init__.py 40
show src/basicts/models/STID/arch/__init__.py 40
show src/basicts/models/STID/arch/stid_arch.py 220
show docs/model_design.md 220
show_class BasicTSModelConfig 120
show_class BasicTSForecastingTaskFlow 200

echo; echo "################ grep: how the graph (adj_mx) is used ################"
grep -rn -i "adj" "$BT/src" 2>/dev/null | head -n 40
echo; echo "################ END ################"
