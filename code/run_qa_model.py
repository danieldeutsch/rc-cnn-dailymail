import qa_model
import utils
import pickle
import json
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, required=True,
                    help='required path to rc train')
parser.add_argument('--dev_path', type=str, required=True,
                    help='required path to rc train')
parser.add_argument('--glove_path', type=str, required=True,
                    help='path to embeddings as required in the original paper. APES uses the 100 dim vectors.')
parser.add_argument('--trained_model_path', type=str, default='./model.pkl.gz',
                    help='Path to trained model')

parser.add_argument('--input_file', required=True, help='The questions to answer')
parser.add_argument('--output_file', required=True, help='The output scores file')

commandline_args = parser.parse_args()

args, word_dict, entity_dict, train_fn, test_fn, params = qa_model.qa_model(train_file=commandline_args.train_path,
                                                                            dev_file=commandline_args.dev_path,
                                                                            embedding_file=commandline_args.glove_path,
                                                                            test_only=True,
                                                                            prepare_model=True,
                                                                            pre_trained=commandline_args.trained_model_path)


def eval_acc(data):
    dev_x1, dev_x2, dev_l, dev_y = utils.vectorize(data, word_dict, entity_dict, verbose=False)
    all_dev = qa_model.gen_examples(dev_x1, dev_x2, dev_l, dev_y, args.batch_size)
    dev_acc, num_correct = qa_model.eval_acc(test_fn, all_dev)
    return dev_acc, num_correct

def read_pickle(file):
	with open(file, 'rb') as f:
		data = pickle.loads(f.read())
	return data

print "*****************************Started answering to questions****************************"

with open(commandline_args.output_file, 'w') as out:
    lines = open(commandline_args.input_file, 'r').read().splitlines()
    for line in tqdm(lines, desc='Answering questions'):
        data = json.loads(line)

        answering_doc = data['answering_doc']
        questioning_doc = data['questioning_doc']
        cands = data['cands']
        doc_questions = data['doc_questions']
        doc_answers = data['doc_answers']
        text = data['text']

        if '@' not in text:
            acc = 0.0
            num_correct = 0
        else:
            acc, num_correct = eval_acc([[text] * len(doc_questions), doc_questions, doc_answers, cands])

        out.write(json.dumps({
            'answering_doc': answering_doc,
            'questioning_doc': questioning_doc,
            'acc': acc,
            'num_correct': num_correct
        }) + '\n')
