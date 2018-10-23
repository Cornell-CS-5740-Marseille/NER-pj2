# Named Entity Recognition with HMMs and MEMMs

In this project, we implement a model that identifies relevant information in a text and tags it with the appropriate label. Particularly, the task of this project is Named Entity Recognition (NER), in which the semantic class associated with a set of contiguous tokens corresponding to a person, location, organization, etc. is annotated.
For this project, We implement Hidden Markov Models and Maximum Entropy Markov Models for Named Entity Recognition (NER) task.

## Getting Started

Our project is implemented by Python 2.7

### Prerequisites

Since we use `nltk` to train the MEMM model, you need to make sure the library has been installed.
If not, you can use pip to install the library.
```
pip install nltk
```

## Project Structure

- **models**: Save the trained models. If you want to train a new model, please delete the previous model.
- **output**: The csv version of Kaggle submissions.
- **Project2_resources**: Requirements, training sets, validation sets and test sets.
- **src**: Codes
    - HMM.py: Implementation of HMM
    - kaggle.py: XXX
    - MEMM.py: Implementation of MEMM
    - ngram.py: N-gram models used by HMM
    - prep.py: Preprocessing of data sets
    - test_baseline.py: XXX
    - test_HMM.py: XXX
    - test_kaggle_HMM.py: Generate Kaggle submissions using HMM
    - test_kaggle_MEMM.py: Generate Kaggle submissions using MEMM
- README.md

## Running the tests

Explain how to run the automated tests for this system

### HMM

The default configuration of the HMM model is currently 
1. No smoothing on transition probability
2. Add-0.01 smoothing on word generation smoothing
3. \<s> and \</s> added to corpus
4. Words not converted to lowercase

These are all configurable in the ```pre_process_hmm()``` method inside ```prep.py```.

To run the test on HMM, simply run
```
python test_HMM.py
```

If you want to change the training file and testing file (currently set to new_train.txt and validation.txt), you can do that inside test_HMM.py.

To generate a Kaggle formatted .csv file, use
```
python test_kaggle_HMM.py
```

### MEMM

You can run the test_kaggle_HMM to generate predicted name entities. 
The model has been trained and saved in the output directory. If you want to train your model, please delete the pickle files and run it again.

```
python test_kaggle_MEMM.py
```
if you want to try the MEMM by your settings, you can:
```
# preprocessing the training data
prepocessing = prep(YOUR_TRAINING_FILE)
data = prepocessing.pre_process_memm()

# train the model
memm_classifier = MEMM(data)
memm_classifier.trainMEMM(True)

# preprocessing the test data
test_prepocessing = prep(YOUR_TEST_FILE')
test_words = test_prepocessing.pre_process_memm_test()

# Viterbi prediction
for sentence in test_words:
    prediction = memm_classifier.viterbi_search(sentence)
```
### Baseline
The baseline system is generated simply by looping through all the words in the training corpus, recording the frequencies of each tag corresponding to the word. When given a  validation file, we look up this dictionary and assign the most frequent tag associated to the word.

To run the test locally on validation data, run
```
python test_baseline.py
```
To generate kaggle competition .csv data, run
```
python test_kaggle_baseline.py
```

## Authors

* **Jialu Li** - *HMM & Documents* - email here
* **Charlie Wang** - *Preprocessing & Baseline* - qw248@cornell.edu
* **Ziyun Wei** - *MEMM* - zw555@cornell.edu


## License

No license

