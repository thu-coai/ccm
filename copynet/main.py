import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
import sys
import json
import math
import os
import time
import random
import sqlite3
random.seed(time.time())
from model import Model, _START_VOCAB

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 30000, "vocabulary size.")
tf.app.flags.DEFINE_integer("num_entities", 21471, "entitiy vocabulary size.")
tf.app.flags.DEFINE_integer("num_relations", 44, "relation size.")
tf.app.flags.DEFINE_integer("embed_units", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("trans_units", 100, "Size of trans embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("copy_use", True, "use copy mechanism or not.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("inference_path", "test", "Set filename of inference, default isscreen")

FLAGS = tf.app.flags.FLAGS
if FLAGS.train_dir[-1] == '/': FLAGS.train_dir = FLAGS.train_dir[:-1]
csk_triples, csk_entities, kb_dict = [], [], []

def prepare_data(path, is_train=True):
    global csk_entities, csk_triples, kb_dict
    
    with open('%s/resource.txt' % path) as f:
        d = json.loads(f.readline())
    
    csk_triples = d['csk_triples']
    csk_entities = d['csk_entities']
    raw_vocab = d['vocab_dict']
    kb_dict = d['dict_csk']
    
    data_train, data_dev, data_test = [], [], []
    
    if is_train:
        with open('%s/trainset.txt' % path) as f:
            for idx, line in enumerate(f):
                #if idx == 100000: break
                if idx % 100000 == 0: print('read train file line %d' % idx)
                data_train.append(json.loads(line))

    with open('%s/validset.txt' % path) as f:
        for line in f:
            data_dev.append(json.loads(line))

    with open('%s/testset.txt' % path) as f:
        for line in f:
            data_test.append(json.loads(line))

    return raw_vocab, data_train, data_dev, data_test

def build_vocab(path, raw_vocab, trans='transE'):
    print("Creating word vocabulary...")
    vocab_list = _START_VOCAB + sorted(raw_vocab, key=raw_vocab.get, reverse=True)
    if len(vocab_list) > FLAGS.symbols:
        vocab_list = vocab_list[:FLAGS.symbols]

    print("Creating entity vocabulary...")
    entity_list = ['_NONE', '_PAD_H', '_PAD_R', '_PAD_T', '_NAF_H', '_NAF_R', '_NAF_T']
    with open('%s/entity.txt' % path) as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity_list.append(e)

    print("Creating relation vocabulary...")
    relation_list = []
    with open('%s/relation.txt' % path) as f:
        for i, line in enumerate(f):
            r = line.strip()
            relation_list.append(r)

    print("Loading word vectors...")
    vectors = {}
    with open('%s/glove.840B.300d.txt' % path) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector
    
    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = map(float, vectors[word].split())
        else:
            vector = np.zeros((FLAGS.embed_units), dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
            
    print("Loading entity vectors...")
    entity_embed = []
    with open('%s/entity_%s.txt' % (path, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            entity_embed.append(map(float, s))

    print("Loading relation vectors...")
    relation_embed = []
    with open('%s/relation_%s.txt' % (path, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            relation_embed.append(s)

    entity_relation_embed = np.array(entity_embed+relation_embed, dtype=np.float32)
    entity_embed = np.array(entity_embed, dtype=np.float32)
    relation_embed = np.array(relation_embed, dtype=np.float32)

    return vocab_list, embed, entity_list, entity_embed, relation_list, relation_embed, entity_relation_embed

def gen_batched_data(data):
    global csk_entities, csk_triples, kb_dict
    encoder_len = max([len(item['post']) for item in data])+1
    decoder_len = max([len(item['response']) for item in data])+1
    triple_len = max([sum([len(tri) for tri in item['all_triples']]) for item in data ])+1
    max_length = 20
    posts, responses, posts_length, responses_length = [], [], [], []
    entities, triples, matches, post_triples, response_triples = [], [], [], [], []
    match_entities, all_entities = [], []
    match_triples, all_triples = [], []
    NAF = ['_NAF_H', '_NAF_R', '_NAF_T']
    PAD = ['_PAD_H', '_PAD_R', '_PAD_T']
    
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)

    def padding_triple(triple, l):
        return [NAF] + triple + [PAD] * (l - len(triple) - 1)

    for item in data:
        posts.append(padding(item['post'], encoder_len))
        responses.append(padding(item['response'], decoder_len))
        posts_length.append(len(item['post'])+1)
        responses_length.append(len(item['response'])+1)
        all_triples.append(padding_triple([csk_triples[x].split(', ') for triple in item['all_triples'] for x in triple], triple_len))
        match_index = []
        for x in item['match_index']:
            _index = [-1] * triple_len
            if x[0] == -1 and x[1] == -1:
                match_index.append(-1)
            else:
                match_index.append(sum([len(m) for m in item['all_triples'][:(x[0]-1)]]) + 1 + x[1])
        match_triples.append(match_index + [-1]*(decoder_len-len(match_index)))

        if not FLAGS.is_train:
            entity = ['_NONE']
            entity += [csk_entities[x] for ent in item['all_entities'] for x in ent]
            entities.append(entity+['_NONE']*(triple_len-len(entity)))


    batched_data = {'posts': np.array(posts),
            'responses': np.array(responses),
            'posts_length': posts_length, 
            'responses_length': responses_length,
            'triples': np.array(all_triples),
            'entities': np.array(entities),
            'match_triples': np.array(match_triples)}
    return batched_data

def train(model, sess, data_train):
    batched_data = gen_batched_data(data_train)
    outputs = model.step_decoder(sess, batched_data, kb_use=True)
    return np.sum(outputs[0])

def generate_summary(model, sess, data_train):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = gen_batched_data(selected_data)
    summary = model.step_decoder(sess, batched_data, kb_use=True, forward_only=True, summary=True)[-1]
    return summary


def evaluate(model, sess, data_dev, summary_writer):
    loss = np.zeros((1, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = gen_batched_data(selected_data)
        outputs = model.step_decoder(sess, batched_data, kb_use=True, forward_only=True)
        loss += np.sum(outputs[0])
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= len(data_dev)
    summary = tf.Summary()
    summary.value.add(tag='decoder_loss/dev', simple_value=loss)
    summary.value.add(tag='perplexity/dev', simple_value=np.exp(loss))
    summary_writer.add_summary(summary, model.global_step.eval())
    print('    perplexity on dev set: %.2f' % np.exp(loss))


def get_steps(train_dir):
    a = os.walk(train_dir)
    for root, dirs, files in a:
        if root == train_dir:
            filenames = files

    steps, metafiles, datafiles, indexfiles = [], [], [], []
    for filename in filenames:
        if 'meta' in filename:
            metafiles.append(filename)
        if 'data' in filename:
            datafiles.append(filename)
        if 'index' in filename:
            indexfiles.append(filename)

    metafiles.sort()
    datafiles.sort()
    indexfiles.sort(reverse=True)

    for f in indexfiles:
        steps.append(int(f[11:-6]))

    return steps

def test(sess, saver, data_dev, setnum=5000):
    with open('%s/stopwords' % FLAGS.data_dir) as f:
        stopwords = json.loads(f.readline())
    steps = get_steps(FLAGS.train_dir)
    low_step = 00000
    high_step = 800000
    with open('%s.res' % FLAGS.inference_path, 'w') as resfile, open('%s.log' % FLAGS.inference_path, 'w') as outfile:
        for step in [step for step in steps if step > low_step and step < high_step]:
            outfile.write('test for model-%d\n' % step)
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, step)
            print('restore from %s' % model_path)
            try:
                saver.restore(sess, model_path)
            except:
                continue
            st, ed = 0, FLAGS.batch_size
            results = []
            loss = []
            while st < len(data_dev):
                selected_data = data_dev[st:ed]
                batched_data = gen_batched_data(selected_data)
                responses, ppx_loss = sess.run(['decoder_1/generation:0', 'decoder/ppx_loss:0'], {'enc_inps:0': batched_data['posts'], 'enc_lens:0': batched_data['posts_length'], 'dec_inps:0': batched_data['responses'], 'dec_lens:0': batched_data['responses_length'], 'entities:0': batched_data['entities'], 'triples:0': batched_data['triples'], 'match_triples:0': batched_data['match_triples']})
                loss += [x for x in ppx_loss]
                for response in responses:
                    result = []
                    for token in response:
                        if token != '_EOS':
                            result.append(token)
                        else:
                            break
                    results.append(result)
                st, ed = ed, ed+FLAGS.batch_size
            match_entity_sum = [.0] * 4
            cnt = 0
            for post, response, result, match_triples, triples, entities in zip([data['post'] for data in data_dev], [data['response'] for data in data_dev], results, [data['match_triples'] for data in data_dev], [data['all_triples'] for data in data_dev], [data['all_entities'] for data in data_dev]):
                setidx = cnt / setnum
                result_matched_entities = []
                triples = [csk_triples[tri] for triple in triples for tri in triple]
                match_triples = [csk_triples[triple] for triple in match_triples]
                entities = [csk_entities[x] for entity in entities for x in entity]
                matches = [x for triple in match_triples for x in [triple.split(', ')[0], triple.split(', ')[2]] if x in response]
                
                for word in result:
                    if word not in stopwords and word in entities:
                        result_matched_entities.append(word)
                outfile.write('post: %s\nresponse: %s\nresult: %s\nmatch_entity: %s\n\n' % (' '.join(post), ' '.join(response), ' '.join(result), ' '.join(result_matched_entities)))
                match_entity_sum[setidx] += len(set(result_matched_entities))
                cnt += 1
            match_entity_sum = [m / setnum for m in match_entity_sum] + [sum(match_entity_sum) / len(data_dev)]
            losses = [np.sum(loss[x:x+setnum]) / float(setnum) for x in range(0, setnum*4, setnum)] + [np.sum(loss) / float(setnum*4)]
            losses = [np.exp(x) for x in losses]
            def show(x):
                return ', '.join([str(v) for v in x])
            outfile.write('model: %d\n\tperplexity: %s\n\tmatch_entity_rate: %s\n%s\n\n' % (step, show(losses), show(match_entity_sum), '='*50))
            resfile.write('model: %d\n\tperplexity: %s\n\tmatch_entity_rate: %s\n\n' % (step, show(losses), show(match_entity_sum)))
            outfile.flush()
            resfile.flush()
    return results

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        raw_vocab, data_train, data_dev, data_test = prepare_data(FLAGS.data_dir)
        vocab, embed, entity_vocab, entity_embed, relation_vocab, relation_embed, entity_relation_embed = build_vocab(FLAGS.data_dir, raw_vocab)
        FLAGS.num_entities = len(entity_vocab)
        print(FLAGS.__flags)
        model = Model(
                FLAGS.symbols, 
                FLAGS.embed_units,
                FLAGS.units, 
                FLAGS.layers,
                embed,
                entity_relation_embed,
                num_entities=len(entity_vocab)+len(relation_vocab),
                num_trans_units=FLAGS.trans_units,
                output_alignments=FLAGS.copy_use)
        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            op_in = model.symbol2index.insert(constant_op.constant(vocab),
                constant_op.constant(range(FLAGS.symbols), dtype=tf.int64))
            sess.run(op_in)
            op_out = model.index2symbol.insert(constant_op.constant(
                range(FLAGS.symbols), dtype=tf.int64), constant_op.constant(vocab))
            sess.run(op_out)
            op_in = model.entity2index.insert(constant_op.constant(entity_vocab+relation_vocab),
                constant_op.constant(range(len(entity_vocab)+len(relation_vocab)), dtype=tf.int64))
            sess.run(op_in)
            op_out = model.index2entity.insert(constant_op.constant(
                range(len(entity_vocab)+len(relation_vocab)), dtype=tf.int64), constant_op.constant(entity_vocab+relation_vocab))
            sess.run(op_out)

        if FLAGS.log_parameters:
            model.print_parameters()

        summary_writer = tf.summary.FileWriter('%s/log' % FLAGS.train_dir, sess.graph)
        loss_step, time_step = np.zeros((1, )), .0
        previous_losses = [1e18]*3
        train_len = len(data_train)
        while True:
            st, ed = 0, FLAGS.batch_size * FLAGS.per_checkpoint
            random.shuffle(data_train)
            while st < train_len:
                start_time = time.time()
                for batch in range(st, ed, FLAGS.batch_size):
                    loss_step += train(model, sess, data_train[batch:batch+FLAGS.batch_size]) / (ed - st)

                show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
                print("global step %d learning rate %.4f step-time %.2f loss %f perplexity %s"
                        % (model.global_step.eval(), model.lr, 
                            (time.time() - start_time) / (ed - st) / FLAGS.batch_size, loss_step, show(np.exp(loss_step))))
                model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, 
                        global_step=model.global_step)
                summary = tf.Summary()
                summary.value.add(tag='decoder_loss/train', simple_value=loss_step)
                summary.value.add(tag='perplexity/train', simple_value=np.exp(loss_step))
                summary_writer.add_summary(summary, model.global_step.eval())
                summary_model = generate_summary(model, sess, data_train)
                summary_writer.add_summary(summary_model, model.global_step.eval())
                evaluate(model, sess, data_dev, summary_writer)
                previous_losses = previous_losses[1:]+[np.sum(loss_step)]
                loss_step, time_step = np.zeros((1, )), .0
                st, ed = ed, min(train_len, ed + FLAGS.batch_size * FLAGS.per_checkpoint)
            model.saver_epoch.save(sess, '%s/epoch/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
    else:
        model = Model(
                FLAGS.symbols, 
                FLAGS.embed_units,
                FLAGS.units, 
                FLAGS.layers,
                embed=None,
                num_entities=FLAGS.num_entities+FLAGS.num_relations,
                num_trans_units=FLAGS.trans_units, 
                output_alignments=FLAGS.copy_use)

        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        print('restore from %s' % model_path)
        model.saver.restore(sess, model_path)
        saver = model.saver

        raw_vocab, data_train, data_dev, data_test = prepare_data(FLAGS.data_dir, is_train=False)

        test(sess, saver, data_test, setnum=5000)
        
