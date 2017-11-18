# fake_news_classifier
## Project for the UE15CS322 Data Analytics Course
### Team members
Hiranmaya Gundu 01FB15ECS127  
Rahul Mayuranath 01FB15ECS225  
Ravi Shreyas Anupindi 01FB15ECS234  

## Credits
Thanks to https://github.com/genyunus/Detecting_Fake_News/ for the cross_validation script.

## Words of Advice:
The models take a fair amount of time to train and test, especially Random Forests and LSTM. This is probably owing to the K-folds cross validation taking quite some time.  
Additionally, to run the models against the entire dataset(30,000 articles) you will require at least 16GB of RAM. To work around this,
navigate to the file **start.py** and make changes to the line:
```python
df = df.sample(1000)
```
This line can be commented out, or you can manually vary the size of the dataset.
For LSTM, navigate to the file **RNN.py** and make changes to the line:
```python
df = df.sample(10000)
```

### Dataset
Link to the dataset: https://drive.google.com/open?id=1Q3ZO3trNfBg0uSlkz73V6XfSRNx7hO7k 
Create a folder called data inside the main directory and add the csv file there.

## Getting started
These instructions will help you in setting up the python environment for running the models and validating results.
After cloning the repository, create a python3 virtual environment to install the required packages.  
```bash
cd fake_news_classifier
python3 -m venv <environment_name>  
source <environment_name>/bin/activate
``` 
This creates the virtual_environment for the project and sets the python interpreter to the project interpreter.
Installing the required packages:
```bash
pip install -r requirements.txt
```
The package also requires additional nltk packages, including the corpus of stop words. These can be downloaded by running the command
```bash
import nltk
nltk.download()
```
from inside the python shell.
