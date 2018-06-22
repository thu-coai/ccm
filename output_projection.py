import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope

def output_projection_layer(num_units, num_symbols, num_samples=None, name="output_projection"):
    def output_fn(outputs):
        return layers.linear(outputs, num_symbols, scope=name)

    def selector_fn(outputs):
        selector = tf.sigmoid(layers.linear(outputs, 1, scope='selector'))
        return selector

    def sequence_loss(outputs, targets, masks):
        with variable_scope.variable_scope('decoder_rnn'):
            logits = layers.linear(outputs, num_symbols, scope=name)
            logits = tf.reshape(logits, [-1, num_symbols])
            local_labels = tf.reshape(targets, [-1])
            local_masks = tf.reshape(masks, [-1])
            
            local_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=local_labels, logits=logits)
            local_loss = local_loss * local_masks
            
            loss = tf.reduce_sum(local_loss)
            total_size = tf.reduce_sum(local_masks)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            
            return loss / total_size

    def sampled_sequence_loss(outputs, targets, masks):
        with variable_scope.variable_scope('decoder_rnn/%s' % name):
            weights = tf.transpose(tf.get_variable("weights", [num_units, num_symbols]))
            bias = tf.get_variable("biases", [num_symbols])
            
            local_labels = tf.reshape(targets, [-1, 1])
            local_outputs = tf.reshape(outputs, [-1, num_units])
            local_masks = tf.reshape(masks, [-1])
            
            local_loss = tf.nn.sampled_softmax_loss(weights, bias, local_labels,
                    local_outputs, num_samples, num_symbols)
            local_loss = local_loss * local_masks
            
            loss = tf.reduce_sum(local_loss)
            total_size = tf.reduce_sum(local_masks)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            
            return loss / total_size
    
    def total_loss(outputs, targets, masks, alignments, triples_embedding, use_entities, entity_targets):
        batch_size = tf.shape(outputs)[0]
        local_masks = tf.reshape(masks, [-1])
        
        logits = layers.linear(outputs, num_symbols, scope='decoder_rnn/%s' % name)
        one_hot_targets = tf.one_hot(targets, num_symbols)
        word_prob = tf.reduce_sum(tf.nn.softmax(logits) * one_hot_targets, axis=2)
        selector = tf.squeeze(tf.sigmoid(layers.linear(outputs, 1, scope='decoder_rnn/selector')))

        triple_prob = tf.reduce_sum(alignments * entity_targets, axis=[2, 3])
        ppx_prob = word_prob * (1 - use_entities) + triple_prob * use_entities
        final_prob = word_prob * (1 - selector) * (1 - use_entities) + triple_prob * selector * use_entities
        final_loss = tf.reduce_sum(tf.reshape( - tf.log(1e-12 + final_prob), [-1]) * local_masks)
        ppx_loss = tf.reduce_sum(tf.reshape( - tf.log(1e-12 + ppx_prob), [-1]) * local_masks)
        sentence_ppx = tf.reduce_sum(tf.reshape(tf.reshape( - tf.log(1e-12 + ppx_prob), [-1]) * local_masks, [batch_size, -1]), axis=1)
        selector_loss = tf.reduce_sum(tf.reshape( - tf.log(1e-12 + selector * use_entities + (1 - selector) * (1 - use_entities)), [-1]) * local_masks)
            
        loss = final_loss + selector_loss
        total_size = tf.reduce_sum(local_masks)
        total_size += 1e-12 # to avoid division by 0 for all-0 weights
        
        return loss / total_size, ppx_loss / total_size, sentence_ppx / tf.reduce_sum(masks, axis=1)



    return output_fn, selector_fn, sequence_loss, sampled_sequence_loss, total_loss
    
