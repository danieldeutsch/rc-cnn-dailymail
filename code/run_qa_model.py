print "running python2 qa_model. don't forget to THEANO_FLAGS=device='cuda1' first"

import qa_model
import utils
import os
import pickle
from time import sleep
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, required=True,
                    help='required path to rc train')
parser.add_argument('--dev_path', type=str, required=True,
                    help='required path to rc train')
parser.add_argument('--queries_folder', type=str, required=True,
                    help='Where to read and write queries.pkl and rewards.txt')
parser.add_argument('--glove_path', type=str, required=True,
                    help='path to embeddings as required in the original paper. APES uses the 100 dim vectors.')
parser.add_argument('--trained_model_path', type=str, default='./model.pkl.gz',
                    help='Path to trained model')

args = parser.parse_args()

query_path = args.queries_folder + '/queries.pkl'
rewards_path = args.queries_folder + '/rewards.txt'

print 'query_path: ' + query_path
print 'rewards_path: ' + rewards_path

args, word_dict, entity_dict, train_fn, test_fn, params = qa_model.qa_model(train_file=args.train_path,
                                                                            dev_file=args.dev_path,
                                                                            embedding_file=args.glove_path,
                                                                            test_only=True,
                                                                            prepare_model=True,
                                                                            pre_trained=args.trained_model_path)


def eval_acc(data):
    dev_x1, dev_x2, dev_l, dev_y = utils.vectorize(data, word_dict, entity_dict)
    all_dev = qa_model.gen_examples(dev_x1, dev_x2, dev_l, dev_y, args.batch_size)
    dev_acc = qa_model.eval_acc(test_fn, all_dev)
    return dev_acc

def read_pickle(file):
	with open(file, 'rb') as f:
		data = pickle.loads(f.read())
	return data

print "*****************************Started answering to questions****************************"

while(True):
    while(not os.path.isfile(query_path)):
        sleep(0.2)
    try:
        data = read_pickle(query_path)
        reward = eval_acc(data[:-1])
        os.remove(query_path)
        rewards_file = open(rewards_path, 'w')
        rewards_file.write(str(reward))
        rewards_file.close()
    except Exception:
        print(query_path)
        print(data)
        print(reward)
        print(str(reward))

