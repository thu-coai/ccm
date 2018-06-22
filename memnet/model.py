import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.layers.python.layers import layers
from dynamic_decoder import dynamic_rnn_decoder
from output_projection import output_projection_layer
from attention_decoder import * 
from tensorflow.contrib.session_bundle import exporter

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
NONE_ID = 0
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class Model(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            embed,
            entity_embed=None,
            num_entities=0,
            num_trans_units=100,
            learning_rate=0.0001,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=60,
            output_alignments=True,
            use_lstm=False):
        
        self.posts = tf.placeholder(tf.string, (None, None), 'enc_inps')  # batch*len
        self.posts_length = tf.placeholder(tf.int32, (None), 'enc_lens')  # batch
        self.responses = tf.placeholder(tf.string, (None, None), 'dec_inps')  # batch*len
        self.responses_length = tf.placeholder(tf.int32, (None), 'dec_lens')  # batch
        self.entities = tf.placeholder(tf.string, (None, None), 'entities')  # batch
        self.entity_masks = tf.placeholder(tf.string, (None, None), 'entity_masks')  # batch
        self.triples = tf.placeholder(tf.string, (None, None, 3), 'triples')  # batch
        self.posts_triple = tf.placeholder(tf.int32, (None, None, 1), 'enc_triples')  # batch
        self.responses_triple = tf.placeholder(tf.string, (None, None, 3), 'dec_triples')  # batch
        self.match_triples = tf.placeholder(tf.int32, (None, None), 'match_triples')  # batch
        encoder_batch_size, encoder_len = tf.unstack(tf.shape(self.posts))
        triple_num = tf.shape(self.triples)[1]
        
        #use_triples = tf.reduce_sum(tf.cast(tf.greater_equal(self.match_triples, 0), tf.float32), axis=-1)
        one_hot_triples = tf.one_hot(self.match_triples, triple_num)
        use_triples = tf.reduce_sum(one_hot_triples, axis=[2])

        self.symbol2index = MutableHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=UNK_ID,
                shared_name="in_table",
                name="in_table",
                checkpoint=True)
        self.index2symbol = MutableHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.string,
                default_value='_UNK',
                shared_name="out_table",
                name="out_table",
                checkpoint=True)
        self.entity2index = MutableHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=NONE_ID,
                shared_name="entity_in_table",
                name="entity_in_table",
                checkpoint=True)
        self.index2entity = MutableHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.string,
                default_value='_NONE',
                shared_name="entity_out_table",
                name="entity_out_table",
                checkpoint=True)
        # build the vocab table (string to index)


        self.posts_word_id = self.symbol2index.lookup(self.posts)   # batch*len
        self.posts_entity_id = self.entity2index.lookup(self.posts)   # batch*len
        #self.posts_word_id = tf.Print(self.posts_word_id, ['use_triples', use_triples, 'one_hot_triples', one_hot_triples], summarize=1e6)
        self.responses_target = self.symbol2index.lookup(self.responses)   #batch*len
        
        batch_size, decoder_len = tf.shape(self.responses)[0], tf.shape(self.responses)[1]
        self.responses_word_id = tf.concat([tf.ones([batch_size, 1], dtype=tf.int64)*GO_ID,
            tf.split(self.responses_target, [decoder_len-1, 1], 1)[0]], 1)   # batch*len
        self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1, 
            decoder_len), reverse=True, axis=1), [-1, decoder_len])
        
        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('word_embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('word_embed', dtype=tf.float32, initializer=embed)
        if entity_embed is None:
            # initialize the embedding randomly
            self.entity_trans = tf.get_variable('entity_embed', [num_entities, num_trans_units], tf.float32, trainable=False)
        else:
            # initialize the embedding by pre-trained word vectors
            self.entity_trans = tf.get_variable('entity_embed', dtype=tf.float32, initializer=entity_embed, trainable=False)

        self.entity_trans_transformed = tf.layers.dense(self.entity_trans, num_trans_units, activation=tf.tanh, name='trans_transformation')
        padding_entity = tf.get_variable('entity_padding_embed', [7, num_trans_units], dtype=tf.float32, initializer=tf.zeros_initializer())

        self.entity_embed = tf.concat([padding_entity, self.entity_trans_transformed], axis=0)

        triples_embedding = tf.reshape(tf.nn.embedding_lookup(self.entity_embed, self.entity2index.lookup(self.triples)), [encoder_batch_size, triple_num, 3 * num_trans_units])
        entities_word_embedding = tf.reshape(tf.nn.embedding_lookup(self.embed, self.symbol2index.lookup(self.entities)), [encoder_batch_size, -1, num_embed_units])


        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts_word_id) #batch*len*unit
        self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_word_id) #batch*len*unit

        encoder_cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])
        decoder_cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])
        
        # rnn encoder
        encoder_output, encoder_state = dynamic_rnn(encoder_cell, self.encoder_input, 
                self.posts_length, dtype=tf.float32, scope="encoder")

        # get output projection function
        output_fn, selector_fn, sequence_loss, sampled_sequence_loss, total_loss = output_projection_layer(num_units, 
                num_symbols, num_samples)

        

        with tf.variable_scope('decoder'):
            # get attention function
            attention_keys_init, attention_values_init, attention_score_fn_init, attention_construct_fn_init \
                    = prepare_attention(encoder_output, 'bahdanau', num_units, imem=triples_embedding, output_alignments=output_alignments)#'luong', num_units)

            decoder_fn_train = attention_decoder_fn_train(
                    encoder_state, attention_keys_init, attention_values_init,
                    attention_score_fn_init, attention_construct_fn_init, output_alignments=output_alignments, max_length=tf.reduce_max(self.responses_length))
            self.decoder_output, _, alignments_ta = dynamic_rnn_decoder(decoder_cell, decoder_fn_train, 
                    self.decoder_input, self.responses_length, scope="decoder_rnn")
            if output_alignments: 
                self.alignments = tf.transpose(alignments_ta.stack(), perm=[1,0,2])
                #self.alignments = tf.Print(self.alignments, [self.alignments], summarize=1e8)
                self.decoder_loss, self.ppx_loss, self.sentence_ppx = total_loss(self.decoder_output, self.responses_target, self.decoder_mask, self.alignments, triples_embedding, use_triples, one_hot_triples)
                self.sentence_ppx = tf.identity(self.sentence_ppx, 'ppx_loss')
                #self.decoder_loss = tf.Print(self.decoder_loss, ['decoder_loss', self.decoder_loss], summarize=1e6)
            else:
                self.decoder_loss, self.sentence_ppx = sequence_loss(self.decoder_output, 
                        self.responses_target, self.decoder_mask)
                self.sentence_ppx = tf.identity(self.sentence_ppx, 'ppx_loss')
         
        with tf.variable_scope('decoder', reuse=True):
            # get attention function
            attention_keys, attention_values, attention_score_fn, attention_construct_fn \
                    = prepare_attention(encoder_output, 'bahdanau', num_units, reuse=True, imem=triples_embedding, output_alignments=output_alignments)#'luong', num_units)
            decoder_fn_inference = attention_decoder_fn_inference(
                    output_fn, encoder_state, attention_keys, attention_values, 
                    attention_score_fn, attention_construct_fn, self.embed, GO_ID, 
                    EOS_ID, max_length, num_symbols, imem=entities_word_embedding, selector_fn=selector_fn)

                
            self.decoder_distribution, _, output_ids_ta = dynamic_rnn_decoder(decoder_cell,
                    decoder_fn_inference, scope="decoder_rnn")
            if output_alignments:
                output_len = tf.shape(self.decoder_distribution)[1]
                output_ids = tf.transpose(output_ids_ta.gather(tf.range(output_len)))
                word_ids = tf.cast(tf.clip_by_value(output_ids, 0, num_symbols), tf.int64)
                entity_ids = tf.reshape(tf.clip_by_value(-output_ids, 0, num_symbols) + tf.reshape(tf.range(encoder_batch_size) * tf.shape(entities_word_embedding)[1], [-1, 1]), [-1])
                entities = tf.reshape(tf.gather(tf.reshape(self.entities, [-1]), entity_ids), [-1, output_len])
                words = self.index2symbol.lookup(word_ids)
                self.generation = tf.where(output_ids > 0, words, entities, name='generation')
            else:
                self.generation_index = tf.argmax(self.decoder_distribution, 2)
                
                self.generation = self.index2symbol.lookup(self.generation_index, name='generation') 
        

        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), 
                trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        self.params = tf.global_variables()
            
        # calculate the gradient of parameters
        #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.lr = opt._lr
       
        gradients = tf.gradients(self.decoder_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)

        tf.summary.scalar('decoder_loss', self.decoder_loss)
        for each in tf.trainable_variables():
            tf.summary.histogram(each.name, each)

        self.merged_summary_op = tf.summary.merge_all()
        
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        
        self.saver_epoch = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000, pad_step_number=True)


    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
    
    def step_decoder(self, session, data, forward_only=False, summary=False, kb_use=True):
        input_feed = {self.posts: data['posts'],
                self.posts_length: data['posts_length'],
                self.responses: data['responses'],
                self.responses_length: data['responses_length'],
                self.triples: data['triples'],
                self.match_triples: data['match_triples']}

        if forward_only:
            output_feed = [self.sentence_ppx]
        else:
            output_feed = [self.sentence_ppx, self.gradient_norm, self.update]
        if summary:
            output_feed.append(self.merged_summary_op)
        return session.run(output_feed, input_feed)
