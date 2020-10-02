mkdir -p dataset
mkdir -p corpus
mkdir -p models/roberta

pip uninstall -y tensorflow
pip install transformers==2.8.0

git config --global user.name "Jordi Mas"
git config --global user.email jmas@softcatala.org
git clone https://github.com/jordimas/bert.git



git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v2.8.0
