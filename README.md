# Blog Posts Classification

Implementation of various machine learning algorithms and deep learning models to predict and classify the age group of blog posts.

Data source: [Blogger](https://www.blogger.com/about/?r=1-null_user)

## Setup
#### Dependencies:
* python3, [tensorflow](https://github.com/tensorflow/tensorflow), [keras](https://github.com/keras-team/keras), [nltk](https://github.com/nltk/nltk), [textstat](https://github.com/shivam5992/textstat), [tqdm](https://github.com/tqdm/tqdm), [scikit-learn](https://github.com/scikit-learn/scikit-learn), etc.

#### To install the dependencies:
```
pip install -r requirements.txt
```

## Usage
### Basic Usage


1. Copy your training and test datasets in the default data directory `../data`

2. Run the command with specific arguments

## Validation script
```
cat ../data/file_name | python3 validation.py [-h] <arguments> 
```
*Example*:
```
cat ../data/train_posts.csv | python3 validation.py --classifier lr \
                                                    --C 0.01 0.05 0.10 \
                                                    --solver newton-cg \
                                                    --max_iter 200
```

#### Arguments
* `--classifier` : Classifier choices = nb: Naive Bayes, lr: Logistic Regression, rf: Random Forest. *Default*: nb
* `--alpha`: Alpha values for classifier=nb. *Default* `[0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]`
* `--C`: C values for classifier=lr. *Default*: `[0.001, 0.1, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 7.5, 10.0]`
* `--test_size`: Fraction of test size. *Default*: `0.10`
* `--max_iter`: Maximum of iterations for classifier=lr.  *Default*: `100`
* `--solver`: Solver for classifier=lr.  *Default*: `lbfgs`. *Choices*: `['lbfgs', 'newton-cg', 'sag', 'saga']`
* `--n_estimators`: Number of estimators for classifier=rf.  *Default*: `300`
* `--max_depth`: Max depth for classifier=rf.  *Default*: `3`

## Prediction script
```
cat ../data/file_name | python3 predictions.py [-h] <arguments> 
```
*Example*:
```
cat ../data/train_posts.csv | python3 predictions.py --classifier rf \
                                                     --n_estimators 200 \
                                                     --max_depth 5 \
                                                     --test_files test_split01.csv test_split02.csv
```

#### Arguments
* `--test_data_dir`: Directory path to the test datasets.  *Default*: `../data`
* `--test_files`: Names of the test files in test_data_dir.  *Default*: `
'test_split01.csv', 'test_split02.csv', 'test_split03.csv', 
'test_split04.csv', 'test_split05.csv', 'test_split06.csv', 'test_split07.csv', 
'test_split08.csv', 'test_split09.csv', 'test_split10.csv', 'test_split11.csv'`
* `--classifier` : Classifier choices = nb: Naive Bayes, lr: Logistic Regression, rf: Random Forest. *Default*: nb
* `--alpha`: Alpha value for classifier=nb. *Default* `0.005`
* `--C`: C value for classifier=lr. *Default*: `7.0`
* `--max_iter`: Maximum of iterations for classifier=lr.  *Default*: `100`
* `--solver`: Solver for classifier=lr.  *Default*: `lbfgs`. *Choices*: `['lbfgs', 'newton-cg', 'sag', 'saga']`
* `--n_estimators`: Number of estimators for classifier=rf.  *Default*: `300`
* `--max_depth`: Max depth for classifier=rf.  *Default*: `3`

## LSTM/BiLSTM train script
```
cat ../data/file_name | python3 lstm_train.py [-h] <arguments> 
```
*Example*:
```
cat ../data/train_posts.csv | python3 lstm_train.py --bilstm \
                                                    --units 32 \
                                                    --spatial_dropout 0.5 \
                                                    --dropout 0.5 \
                                                    --recurrent_dropout 0.5 \
                                                    --batch_size 256 \
                                                    --epochs 5
```

#### Arguments
* `--test_size`: Fraction of test size.  *Default*: `0.10`
* `--vocab_size`: Vocabulary size for the embedding layer.  *Default*: `10000`
* `--max_words` : Maximum number of words per blog. *Default*: `250`
* `--embedding_dim`: Dimension of the embedding. *Default* `100`
* `--bilstm`: Boolean indicating to use BiLSTM instead of LSTM.
* `--units`: Number of units in the LSTM layer.  *Default*: `100`
* `--spatial_dropout`: Spatial dropout 1D.  *Default*: `0.4`
* `--dropout`: Dropout of the LSTM layer.  *Default*: `0.4`
* `--recurrent_dropout`: Recurrent dropout of the LSTM layer.  *Default*: `0.4`
* `--batch_size`: Batch size.  *Default*: `1000`
* `--epochs`: Number of epochs.  *Default*: `10`

## LSTM/BiLSTM prediction script
```
cat ../data/file_name | python3 lstm_predict.py [-h] <arguments> 
```
*Example*:
```
cat ../data/train_posts.csv | python3 lstm_predict.py --bilstm \
                                                      --units 32 \
                                                      --spatial_dropout 0.5 \
                                                      --dropout 0.5 \
                                                      --recurrent_dropout 0.5 \
                                                      --batch_size 256 \
                                                      --epochs 5 \
                                                      --test_files test_split01.csv test_split02.csv
```

#### Arguments
* `--test_data_dir`: Directory path to the test datasets.  *Default*: `../data`
* `--test_files`: Names of the test files in test_data_dir.  *Default*: `
'test_split01.csv', 'test_split02.csv', 'test_split03.csv', 
'test_split04.csv', 'test_split05.csv', 'test_split06.csv', 'test_split07.csv', 
'test_split08.csv', 'test_split09.csv', 'test_split10.csv', 'test_split11.csv'`
* `--vocab_size`: Vocabulary size for the embedding layer.  *Default*: `10000`
* `--max_words` : Maximum number of words per blog. *Default*: `250`
* `--embedding_dim`: Dimension of the embedding. *Default* `100`
* `--bilstm`: Boolean indicating to use BiLSTM instead of LSTM.
* `--units`: Number of units in the LSTM layer.  *Default*: `100`
* `--spatial_dropout`: Spatial dropout 1D.  *Default*: `0.4`
* `--dropout`: Dropout of the LSTM layer.  *Default*: `0.4`
* `--recurrent_dropout`: Recurrent dropout of the LSTM layer.  *Default*: `0.4`
* `--batch_size`: Batch size.  *Default*: `1000`
* `--epochs`: Number of epochs.  *Default*: `1`

## Dummy classifier

#### Create the model
cat ../data/train_posts.csv | python3 dummy_model.py

#### Prediction
cat ../data/test_split01.csv | python3 dummy_predict.py models/dummy-most.clf

#### Evaluate
cat ../data/test_split01.csv | python3 dummy_eval.py out/dummy-most.clf.out


## Universal Sentence Encoder with Google Colab IPython Notebook

1. Open IFT6285-Dev1.ipynb in *Google Colab*
2. Upload the test splits to the `My Drive/Colab Notebooks/` repository
3. Activate the GPU by selecting Runtime > Change runtime type > Hardware accelerator > GPU > save *(optional)*
4. Run the notebook



# Authors 
Thach Jean-Pierre *- University of Montreal*

Wong Leo *- University of Montreal*