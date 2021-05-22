# Deep Learning for Sentiment Analysis
Project guidlines: [link](http://faculty.cooper.edu/sable2/courses/spring2021/ece467/NLP_Spring21_PS3.docx)

Original dataset: [dataset](https://www.kaggle.com/kazanova/sentiment140)

## TLDR:
* Use any deep learning framework to implement a deep learning NLP project

## File Breakdown:
* **dataset.py -** Takes the input dataset from kaggle and converts it into the format needed for the deep learning model (drop neutral labels and converts positive labels to 1's). 
  Also performs some preprocessing (removing emails, tags, stop words, & lemmatizes input). 
* **data.py -** Data processing class. Reads list or file, optionally splits data, and converts to TensorDataset. 
* **prepro.py -** Preprocessing class. Handles padding, encoding, & decoding. 
* **model.py -** Defines the deep learning model. 
* **train.py -** Training loop, trains a model (optionally concurrently validating on a second dataset)
* **test.py -** Test trained model on an external dataset
* **baseline.py -** Calculates the baseline accuracy for the model on a given dataset (always pick the most likely label). 

## Usage:
```
dataset.py [-h] [--data_path DATA_PATH] [--size SIZE] [--test_split TEST_SPLIT]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        dataset file path
  --size SIZE           total number of tweets in final dataset
  --test_split TEST_SPLIT
                        fraction of dataset to save for test set

```
```
train.py [-h] [--train_path TRAIN_PATH] [--validation_count VALIDATION_COUNT] [--batch_size BATCH_SIZE] [--lr LR] 
                [--epochs EPOCHS] [--max_vocab MAX_VOCAB] [--embedding_dim EMBEDDING_DIM] [--hidden_dim HIDDEN_DIM]
                [--model_save_path MODEL_SAVE_PATH] [--prepro_save_path PREPRO_SAVE_PATH]


optional arguments:
  -h, --help            show this help message and exit
  --train_path TRAIN_PATH
                        training file path
  --validation_count VALIDATION_COUNT
                        number of inputs to save for validation
  --batch_size BATCH_SIZE
                        batch size
  --lr LR               learning rate
  --epochs EPOCHS       number of epochs to train
  --max_vocab MAX_VOCAB
                        maximum vocab size
  --embedding_dim EMBEDDING_DIM
                        embedding dimension size
  --hidden_dim HIDDEN_DIM
                        hidden layer size
  --model_save_path MODEL_SAVE_PATH
                        file path for saved model
  --prepro_save_path PREPRO_SAVE_PATH
                        file path for saved preprocessor
```
```
test.py [-h] [--test_path TEST_PATH] [--max_vocab MAX_VOCAB] [--model_path MODEL_PATH] [--prepro_path PREPRO_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --test_path TEST_PATH
                        training file path
  --max_vocab MAX_VOCAB
                        maximum vocab size
  --model_path MODEL_PATH
                        path to trained model
  --prepro_path PREPRO_PATH

```
```
baseline.py [-h] [--train_path TRAIN_PATH] [--test_path TEST_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --train_path TRAIN_PATH
                        training file path
  --test_path TEST_PATH
                        testing file path

```
