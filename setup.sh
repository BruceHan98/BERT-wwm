export SYNONYMS_WORD2VEC_BIN_URL_ZH_CN=https://gitee.com/chatopera/cskefu/attach_files/610602/download/words.vector.gz
pip install -U synonyms
python -c "import synonyms"

pip install tensorflow-gpu==1.15
pip install transformers==4.10.0