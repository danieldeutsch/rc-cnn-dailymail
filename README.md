# CNN/Daily Mail Reading Comprehension Task

This is a fork of https://github.com/danqi/rc-cnn-dailymail. code for [A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task](https://arxiv.org/pdf/1606.02858v2.pdf).
For exaplnation go to the [original repository](https://github.com/danqi/rc-cnn-dailymail).

This fork is to enable quering QA on a trained model. A trained model over the [CNN dataset](http://cs.stanford.edu/~danqi/data/cnn.tar.gz) is available [here](https://github.com/mataney/rc-cnn-dailymail/blob/master/code/model.pkl.gz).

## Dependencies
* Python 2.7
* Theano >= 0.7
* Lasagne 0.2.dev1

## Train model
For more explanation about training a model go to the [original repository](https://github.com/danqi/rc-cnn-dailymail).

## Running
When running `python code/run_qa_model.py --folder folder_path ..` you start a QA stream that expects questions in `folder_path/queries.pkl` and returns its rewarding accuracy in `folder_path/rewards.txt`

Also required to pass:

`--train_path --dev_path --glove_path`
