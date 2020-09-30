mkdir -p dataset
mkdir -p corpus
mkdir -p models/roberta

pip install tensorflow==2.1.0
pip install transformers==2.5.0 # in sync with run_language_modeling
pip install tokenizers==0.5.0
pip install torch==1.6.0
pip install tensorboard==2.1.0

git config --global user.name "Jordi Mas"
git config --global user.email jmas@softcatala.org
git clone https://github.com/jordimas/bert.git



git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v2.5.0
