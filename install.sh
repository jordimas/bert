mkdir -p data
mkdir -p models/roberta

pip uninstall -y tensorflow
pip install transformers==2.8.0
pip install torch==1.6.0
pip install tensorboard==2.1.0


git config --global user.name "Jordi Mas"
git config --global user.email jmas@softcatala.org
git clone https://github.com/jordimas/bert.git
sudo updatedb &


#git clone https://github.com/huggingface/transformers.git
#cd transformers
#git checkout v2.8.0
