# coding:utf-8
import collections
poems=['123','123','13']
all_words=['123','23','3','232','23','23','123']
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)

# 取前多少个常用字
words = words[:len(words)] + (' ',)
#每个字映射为一个数字ID,使用word2vec将字映射效果会更好
word_int_map = dict(zip(words, range(len(words))))
poems_vector = [list(map(lambda word: word_int_map.get(word, len(words)), poem)) for poem in poems]
print(poems_vector)
print('words',words)
print('wrod_int_map',word_int_map)
