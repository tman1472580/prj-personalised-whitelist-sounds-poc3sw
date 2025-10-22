Use cases with Python 3.10 (3.13 keeps having dependecies issues):

python3 -m venv whitelist

python -m pip install -r requirements.txt

for training run:
python yamnet_256_training.py
Just change the dataset directory on root directory

for inferencing: 
this loads however many files you want to test on from the dataset and also compares performance
to original yamnet:.ue
python all_yamnet_infrences.py  --num-samples 200 --compare-original


