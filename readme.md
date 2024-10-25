# Diamond Price Predictor


## Installation
### Step 1:

Download the repository

```console
git clone && cd diamond-predictor
```

### Step 2:


Initialise the python virtual environment

``` console
python -m venv venv
```

### Step 3

Download the dependancies

```consol
pip intall -r requirements.txt
```

### Step 4
Select a model to use

| Model                   | Accuracy | File                       |
| ----------------------- | -------- | -------------------------- |
| Random Forest Regressor | 87%      | random_forest.py           |
| Decision Tree           | 83%      | decision_tree_regressor.py |
| Linear Regressor        | 77%      | linear_regression.py       |
| Ada Boost               | 63%      | adaboost.py                |
| SVM Regressor           | 58%      | svm_regressor.py           |
| Neural Network          | 37%      | neural_network.py          |


## Step 5
Run the server
```console
python server.py
```
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000)

