import numpy as np
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy_step(nn.Module):
    def __init__(self, m, embedding_size, hidden_size):
        super(Policy_step, self).__init__()
        self.batch_norm = nn.BatchNorm1d(m * hidden_size)
        self.lstm_cell = nn.LSTMCell(input_size=2 * m * embedding_size, hidden_size=2 * m * hidden_size)
        self.l1 = nn.Linear(m * embedding_size, 2 * m * embedding_size)
        self.l2 = nn.Linear(2 * m * embedding_size, m * embedding_size)
        self.l3 = nn.Linear(2 * m * embedding_size, m * embedding_size)

    def forward(self, prev_action, prev_state):
        prev_action = torch.relu(self.l1(prev_action))
        output, ch = self.lstm_cell(prev_action, prev_state)
        output = torch.relu(self.l2(output))
        ch = torch.relu(self.l3(ch))
        ch = torch.cat([output.unsqueeze(0).unsqueeze(0), ch.unsqueeze(0).unsqueeze(0)], dim=1)
        return output, ch


class Policy_mlp(nn.Module):
    def __init__(self, hidden_size, m, embedding_size):
        super(Policy_mlp, self).__init__()
        self.hidden_size = hidden_size
        self.m = m
        self.embedding_size = embedding_size
        self.mlp_l1 = nn.Linear(2 * m * self.hidden_size, m * self.hidden_size, bias=True)
        self.mlp_l2 = nn.Linear(m * self.hidden_size, m * self.embedding_size, bias=True)

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = torch.relu(self.mlp_l2(hidden))
        return output


class Agent(nn.Module):
    def __init__(self, params):
        super(Agent, self).__init__()
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = params['entity_vocab']['PAD']
        self.rPAD = params['relation_vocab']['PAD']

        self.use_entity_embeddings = params['use_entity_embeddings']
        self.train_entity_embeddings = params['train_entity_embeddings']
        self.train_relation_embeddings = params['train_relation_embeddings']


        # if params['use_entity_embeddings']:
        #     self.entity_initializer = tf.contrib.layers.xavier_initializer()
        # else:
        #     self.entity_initializer = tf.zeros_initializer()
        # self.train_entities = params['train_entity_embeddings']
        # self.train_relations = params['train_relation_embeddings']

        if self.train_entity_embeddings:
            self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size)
        else:
            self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size).requires_grad_(
                False)
        torch.nn.init.xavier_uniform_(self.entity_embedding.weight)

        if self.train_relation_embeddings:
            self.relation_embedding = nn.Embedding(self.action_vocab_size, 2 * self.embedding_size)
        else:
            self.relation_embedding = nn.Embedding(self.action_vocab_size, 2 * self.embedding_size).requires_grad_(
                False)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)
        

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        # self.dummy_start_label = tf.constant(
        #     np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])
        self.dummy_start_label = (torch.ones(self.batch_size) * params['relation_vocab']['DUMMY_START_RELATION']).long()

        self.entity_embedding_size = self.embedding_size

        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2
        
        self.policy_step = Policy_step(m=self.m, embedding_size=self.embedding_size, hidden_size=self.hidden_size)
        self.policy_mlp = Policy_mlp(self.hidden_size, self.m, self.embedding_size)

        self.gate1_linear = nn.Linear(2*self.hidden_size, 3*2*self.hidden_size)
        self.gate2_linear = nn.Linear(2*self.hidden_size, 3*2*self.hidden_size)

        # with tf.variable_scope("action_lookup_table"):
        #     self.action_embedding_placeholder = tf.placeholder(tf.float32,
        #                                                        [self.action_vocab_size, 2 * self.embedding_size])

        #     self.relation_lookup_table = tf.get_variable("relation_lookup_table",
        #                                                  shape=[self.action_vocab_size, 2 * self.embedding_size],
        #                                                  dtype=tf.float32,
        #                                                  initializer=tf.contrib.layers.xavier_initializer(),
        #                                                  trainable=self.train_relations)
        #     self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        # with tf.variable_scope("entity_lookup_table"):
        #     self.entity_embedding_placeholder = tf.placeholder(tf.float32,
        #                                                        [self.entity_vocab_size, 2 * self.embedding_size])
        #     self.entity_lookup_table = tf.get_variable("entity_lookup_table",
        #                                                shape=[self.entity_vocab_size, 2 * self.entity_embedding_size],
        #                                                dtype=tf.float32,
        #                                                initializer=self.entity_initializer,
        #                                                trainable=self.train_entities)
        #     self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

        # with tf.variable_scope("policy_step"):
        #     cells = []
        #     for _ in range(self.LSTM_Layers):
        #         cells.append(tf.contrib.rnn.LSTMCell(self.m * self.hidden_size, use_peepholes=True, state_is_tuple=True))
        #     self.policy_step = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    # def policy_MLP(self, state):
    #     with tf.variable_scope("MLP_for_policy"):
    #         hidden = tf.layers.dense(state, 4 * self.hidden_size, activation=tf.nn.relu)
    #         output = tf.layers.dense(hidden, self.m * self.embedding_size, activation=tf.nn.relu)
    #     return output

    # def action_encoder(self, next_relations, next_entities):
    #     with tf.variable_scope("lookup_table_edge_encoder"):
    #         relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
    #         entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
    #         if self.use_entity_embeddings:
    #             action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
    #         else:
    #             action_embedding = relation_embedding
    #     return action_embedding
    
    def action_encoder(self, next_relations, next_entities):
        relation_embedding = self.relation_embedding(next_relations)
        entity_embedding = self.entity_embedding(next_entities)
        if self.use_entity_embeddings:
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        else:
            action_embedding = relation_embedding
        return action_embedding

    def step(self, next_relations, next_entities, prev_state, prev_relation, query_embedding, current_entities,
             range_arr, first_step_of_test):

        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        # 1. one step of rnn
        prev_state = torch.squeeze(prev_state, dim=0)
        output, new_state = self.policy_step(prev_action_embedding, (prev_state[0], prev_state[1]))  # output: [B, 4D]

        # Get state vector
        # prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table, current_entities)
        prev_entity = self.entity_embedding(current_entities)
        if self.use_entity_embeddings:
            state = torch.cat([output, prev_entity], dim=-1)
        else:
            state = output
        # if self.use_entity_embeddings:
        #     state = tf.concat([output, prev_entity], axis=-1)
        # else:
        #     state = output
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        state_query_concat = torch.cat([state, query_embedding], dim=-1)

        # MLP for policy#

        output = self.policy_MLP(state_query_concat)
        output_expanded = torch.unsqueeze(output, dim=1)  # [B, 1, 2D]
        # output_expanded = tf.expand_dims(output, axis=1)  # [B, 1, 2D]
        # prelim_scores = tf.reduce_sum(tf.multiply(candidate_action_embeddings, output_expanded), axis=2)
        prelim_scores = torch.sum(candidate_action_embeddings * output_expanded, dim=2)

        # Masking PAD actions

        # comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD  # matrix to compare
        comparison_tensor = torch.ones_like(next_relations).int() * self.rPAD
        # mask = tf.equal(next_relations, comparison_tensor)  # The mask
        mask = next_relations == comparison_tensor
        # dummy_scores = tf.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        dummy_scores = torch.ones_like(prelim_scores) * -99999.0
        # scores = tf.where(mask, dummy_scores, prelim_scores)  # [B, MAX_NUM_ACTIONS]
        scores = torch.where(mask, dummy_scores, prelim_scores)
        # 4 sample action
        # action = tf.to_int32(tf.multinomial(logits=scores, num_samples=1))  # [B, 1]
        action = torch.distributions.categorical.Categorical(logits=scores)
        label_action = action.sample()
        # loss
        # 5a.
        # label_action =  tf.squeeze(action, axis=1)
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)  # [B,]
        loss = torch.nn.CrossEntropyLoss(reduce=False)(scores, label_action)
        # 6. Map back to true id
        # action_idx = tf.squeeze(action)
        # chosen_relation = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))
        chosen_relation = next_relations[list(torch.stack([range_arr, label_action]))]
        return loss, new_state, F.log_softmax(scores), label_action, chosen_relation

    def __call__(self, candidate_relation_sequence, candidate_entity_sequence, current_entities,
                 path_label, query_relation, range_arr, first_step_of_test, T=3, entity_sequence=0):

        self.baseline_inputs = []
        # get the query vector
        query_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, query_relation)  # [B, 2D]
        state = self.policy_step.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        prev_relation = self.dummy_start_label

        all_loss = []  # list of loss tensors each [B,]
        all_logits = []  # list of actions each [B,]
        action_idx = []

        with tf.variable_scope("policy_steps_unroll") as scope:
            for t in range(T):
                if t > 0:
                    scope.reuse_variables()
                next_possible_relations = candidate_relation_sequence[t]  # [B, MAX_NUM_ACTIONS, MAX_EDGE_LENGTH]
                next_possible_entities = candidate_entity_sequence[t]
                current_entities_t = current_entities[t]

                path_label_t = path_label[t]  # [B]

                loss, state, logits, idx, chosen_relation = self.step(next_possible_relations,
                                                                              next_possible_entities,
                                                                              state, prev_relation, query_embedding,
                                                                              current_entities_t,
                                                                              label_action=path_label_t,
                                                                              range_arr=range_arr,
                                                                              first_step_of_test=first_step_of_test)

                all_loss.append(loss)
                all_logits.append(logits)
                action_idx.append(idx)
                prev_relation = chosen_relation

            # [(B, T), 4D]

        return all_loss, all_logits, action_idx
