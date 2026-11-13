#!/bin/bash
# Runs the Week 8/9 tutorial (Steps 1 to 11) and the two Homework 3 scripts.
# Every step's printed output is saved in the logs/ folder.
cd "$(dirname "$0")" || exit 1
mkdir -p logs data models texts
exec > >(tee logs/00_run_all.txt) 2>&1
echo "Started: $(date)"

# 1. Python environment, kept OUTSIDE OneDrive so thousands of package files do not sync
VENV="$HOME/.venvs/ids570_hw3"
if [ ! -x "$VENV/bin/python" ]; then "$HOME/miniforge3/bin/python3" -m venv "$VENV"; fi
"$VENV/bin/pip" install -q nltk gensim scikit-learn tqdm joblib
if ! "$VENV/bin/python" -c "import gensim, nltk, sklearn" 2>/dev/null; then
  echo "Packages did not install on this Python. Trying a Python 3.12 environment with conda."
  VENV="$HOME/.venvs/ids570_hw3_py312"
  [ -x "$VENV/bin/python" ] || "$HOME/miniforge3/bin/conda" create -y -q -p "$VENV" python=3.12
  "$VENV/bin/pip" install -q nltk gensim scikit-learn tqdm joblib
fi
PYTHON="$VENV/bin/python"
"$PYTHON" -c "import sys, sklearn, nltk, gensim; print('Python', sys.version.split()[0], '| scikit-learn', sklearn.__version__, '| nltk', nltk.__version__, '| gensim', gensim.__version__)" | tee logs/00_versions.txt
"$PYTHON" -c "import gensim, nltk, sklearn" || { echo "FAILED packages" > logs/STATUS.txt; exit 1; }

# 2. Canvas text files into texts/
if [ "$(ls texts/*.txt 2>/dev/null | wc -l)" -lt 500 ]; then
  curl -L --fail -sS -w "HTTP %{http_code} %{size_download} bytes\n" -o Train_Test_Documents.zip "$(cat canvas_zip_url.txt)" || { echo "FAILED download" > logs/STATUS.txt; exit 1; }
  unzip -q -o -j Train_Test_Documents.zip '*.txt' -d texts
  rm -f Train_Test_Documents.zip canvas_zip_url.txt
fi
echo "Text files in texts/: $(ls texts/*.txt | wc -l)"

# 3. Run every step in order
for s in step1_load_texts step2_segment_and_chunk step3_tokenize_chunks step4_process_all_files \
         step5_train_word2vec step5b_query_word2vec step6_define_tiers step7_label_chunks \
         step9_prepare_datasets step10_tfidf_representation step11_train_classifier \
         hw03_l2_baseline hw03_l1_model; do
  echo "--- $(date +%H:%M:%S) running $s.py"
  echo "RUNNING $s" > logs/STATUS.txt
  "$PYTHON" "$s.py" > "logs/$s.txt" 2>&1 || { echo "FAILED $s" > logs/STATUS.txt; echo "FAILED $s"; exit 1; }
done
echo "DONE" > logs/STATUS.txt
echo "Finished: $(date). You can close this window."
