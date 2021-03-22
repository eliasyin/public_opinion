import stanza
import numpy as np
import networkx as nx

class DepParse:
    def __init__(self):
        self.rel_vocab = {  
            # 0表示无关
            'amod' : 1,
            'advmod' : 2,
            'nsubj' : 3,
            'conj' : 4,
            'ccomp' : 5,

            # 需要忽略的一些依赖关系
            'punct' : 0,
            'det ' : 0
        }
        self.pos_vocab = {
            'JJ' : 1
        }


    def get_parser(self):
        config = {
            'lang' : 'zh',
            'processors' : 'tokenize, pos, lemma, depparse',
            'tokenize_pretokenized' : True
        }
        nlp = stanza.Pipeline(**config)
        return nlp

    def get_adjs(self, nlp, raw_tokens):
        """AI is creating summary for get_adjs

        Args:
            nlp ([type]): [description]
            tokens ([list]): format:
                                ['The', 'bread', 'is', 'top', 'notch', 'as', 'well', '.']

        Raises:
            ValueError: [description]
        """
        raw_tokens = ' '.join(raw_tokens).split('[SEP]')[0].split()
        doc = nlp([raw_tokens])  # expand the dim of tokens
        sent = doc.sentences[0]
        nlp_tokens = [word.text for word in sent.words]
        if len(nlp_tokens) != len(raw_tokens):
            raise ValueError("Length of stanza tokens does not equal to pre-tokens")
        dep_adj = np.eye(len(raw_tokens), dtype='float32') 
        adj_mask = np.ones((len(raw_tokens), len(raw_tokens)), dtype='float32')
        dep_rel, dep_rel_head, pos_tag = [], [], []
        for word in sent.words:
            pos_tag.append(word.pos)
            dep_rel.append(word.deprel)
            dep_rel_head.append(word.head - 1)
            if word.head != 0:
                adj_tail = word.id - 1
                adj_head = word.head - 1
                dep_adj[adj_head][adj_tail] = 1
        self._update_vocab(dep_rel, pos_tag)
        rel_adj = self._cal_rel_adj(raw_tokens, dep_rel_head, dep_rel)
        pos_tag_idx = [self.pos_vocab[tag] for tag in pos_tag]
        # pad to the size of bert base input
        # dep_adj = np.pad(dep_adj, 1, 'constant')
        # rel_adj = np.pad(rel_adj, 1, 'constant')
        # adj_mask = np.pad(adj_mask, 1, 'constant')
        return dep_adj, rel_adj, adj_mask, pos_tag_idx




    def _update_vocab(self, dep_rel, pos_tag):
        # update rel vocab
        idx = max(list(self.rel_vocab.values())) + 1
        for rel in dep_rel:
            if rel not in list(self.rel_vocab.keys()):
                self.rel_vocab[rel] = idx
                idx = idx + 1
        # update pos tag vocab
        idx = max(list(self.pos_vocab.values())) + 1
        for tag in pos_tag:
            if tag not in list(self.pos_vocab.keys()):
                self.pos_vocab[tag] = idx
                idx = idx + 1


    def _cal_rel_adj(self, all_tokens, dep_rel_head, dep_rel):
        length = len(all_tokens)
        dep_rel_adj = np.zeros((length, length), dtype='float32')
        for tail in range(length):
            head = dep_rel_head[tail] - 1
            rel_tag = dep_rel[tail]
            dep_rel_adj[head][tail] = self.rel_vocab[rel_tag]
        return dep_rel_adj

    def calculate_dep_dist(self, nlp, raw_tokens, aspect_tokens):
        r'''
        根据句法解析树计算各个token到aspect的最短距离
        input : 
        -----------
        sentence : 完整句子
        aspect : aspect
        '''
        raw_tokens = ' '.join(raw_tokens).split('[SEP]')[0].split()
        doc = nlp([raw_tokens])  # expand the dim of tokens
        sent = doc.sentences[0]
        nlp_tokens = [word.text for word in sent.words]
        if len(nlp_tokens) != len(raw_tokens):
            raise ValueError("Length of stanza tokens does not equal to pre-tokens")
        edges = []
        for word in sent.words:
            edges.append((word.head, word.id))

        graph = nx.Graph(edges)

        dist = [0.0] * len(raw_tokens)
        for idx in range(len(raw_tokens)):
            distance = [0.0] * len(aspect_tokens)
            for i, aspect_token in enumerate(aspect_tokens):
                aspect_id = raw_tokens.index(aspect_token)+1
                distance[i]= len(nx.shortest_path(graph, source=edges[idx][1], 
                                                            target=aspect_id, weight=1))-1
            dist[idx]  = min(distance)
        return dist
