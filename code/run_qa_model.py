print "running python2 qa_model. don't forget to THEANO_FLAGS=device='cuda1' first"

import qa_model
import utils
import os
import pickle
from time import sleep
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--run_id', type=int,
                    help='run id to append to the query and reward files')
parser.add_argument('--folder', type=int, required=True, choices=[0,1,2, 3],
                    help='if this is used to justify APES on TAC')

args = parser.parse_args()
print 'run_id: '+str(args.run_id)

if args.folder == 0:
    query_path = '/matan_files/OpenNMT-py/queries'+str(args.run_id)+'.pkl'
    rewards_path = '/matan_files/OpenNMT-py/rewards'+str(args.run_id)+'.txt'
elif args.folder == 1:
    query_path = '/matan_files/justifying_APES_on_TAC2011/queries'+str(args.run_id)+'.pkl'
    rewards_path = '/matan_files/justifying_APES_on_TAC2011/rewards'+str(args.run_id)+'.txt'
elif args.folder == 2:
    query_path = '/matan_files/OpenNMT-py-EMNLP/testout/grid_search_over_alpha_beta_gamma/apes_scores/queries'+str(args.run_id)+'.pkl'
    rewards_path = '/matan_files/OpenNMT-py-EMNLP/testout/grid_search_over_alpha_beta_gamma/apes_scores/rewards'+str(args.run_id)+'.txt'
else:
    query_path = '/matan_files/OpenNMT-py-EMNLP/my_scripts/queries'+str(args.run_id)+'.pkl'
    rewards_path = '/matan_files/OpenNMT-py-EMNLP/my_scripts/rewards'+str(args.run_id)+'.txt'

print 'query_path: ' + query_path
print 'rewards_path: ' + rewards_path

model_path = '/matan_files/rc-cnn-dailymail/code/model.pkl.gz'
args, word_dict, entity_dict, train_fn, test_fn, params = qa_model.qa_model(train_file='/matan_files/datasets/cnn/cnn_qa/train.txt', dev_file='/matan_files/datasets/cnn/cnn_qa/test.txt', embedding_file='/matan_files/word-embeddings/glove.6B.100d.txt', test_only=True, prepare_model=True, pre_trained=model_path)

#model_path = '/matan_files/manning_trained_models/daily-mail-model.pkl.gz'
#args, word_dict, entity_dict, train_fn, test_fn, params = qa_model.qa_model(train_file='/matan_files/datasets/dailymail/dailymail_qa/train.txt', dev_file='/matan_files/datasets/dailymail/dailymail_qa/test.txt', embedding_file='/matan_files/word-embeddings/glove.6B.100d.txt', test_only=True, prepare_model=True, pre_trained=model_path)


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
        reward = eval_acc(data[:-1]) #I don't use the cands
        os.remove(query_path)
        rewards_file = open(rewards_path, 'w')
        rewards_file.write(str(reward))
        rewards_file.close()
    except Exception:
        print(query_path)
        print(data)
        print(reward)
        print(str(reward))

