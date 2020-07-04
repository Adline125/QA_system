import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics.pairwise import cosine_similarity
import heapq


# 读入语料
def read_corpus():
    qlist = []
    alist = []
    with open('train-v2.0.json', 'r') as f:
        load_dict = json.load(f)
        print(type(load_dict))
        for data in load_dict['data']:
            for paragraph in data['paragraphs']:
                for qa in paragraph['qas']:
                    question = qa['question']
                    qlist.append(question)
                    if len(qa['answers']) == 0:
                        alist.append(qa['plausible_answers'][0]['text'])
                    else:
                        alist.append(qa['answers'][0]['text'])
    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, alist

# 语料预处理
s_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
def sent_preprocess(q):
    words = q.split(' ')
    new_q = ''
    for word in words:
        if word not in s_words:  # 1. 停用词过滤
            if re.match('[1234567890]+$', word):  # 5. 数字的处理
                word = '#number'
            word =lemmatizer.lemmatize(word, 'n')  # 6. lemmazation
            word =lemmatizer.lemmatize(word, 'v')  # 6. lemmazation
            word =lemmatizer.lemmatize(word, 'a')  # 6. lemmazation
            new_q += word.lower().replace('?', '').replace('!', '').replace(',', '').replace('.', '') + ' '  # 2. 转换成lower_case  3. 去掉一些无用的符号
    return new_q.strip()


# 寻找语料中词的相关词
# 读语料及预处理
qlist, alist = read_corpus()
temp = []
for q in qlist:
    new_q = sent_preprocess(q)
    temp.append(new_q)
qlist.clear()
del qlist
qlist = temp

with open('related_words.txt', 'w') as file:
    # 加载训练语料中经预处理后的所有词
    all_words = set()
    for q in qlist:
        for word in q.split(' '):
            all_words.add(word)
    all_words = list(all_words)

    # 加载glove词向量
    emb_dic = {}
    with open('glove.6B.200d.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            emb_dic[line[0: line.index(' ')]] = line[line.index(' ') + 1:].split()

    # 计算相似词列表并写入文件
    # emb_size = len(list(emb_dic.values())[0])
    cnt = 0  # 计数，方便查看进度
    for word in all_words:
        cnt += 1
        w_embed = emb_dic.get(word)
        if not w_embed:
            continue
        cnt2 = 0  # 计数，方便查看进度
        tasks = []
        for idx, cand in enumerate(all_words):
            cnt2 += 1
            print(f"{cnt} + ':' + {cnt2}")
            if word == cand:
                continue
            c_embed = emb_dic.get(cand)
            if not c_embed:
                continue
            sim = cosine_similarity([w_embed, c_embed])
            sim_r = sim[0][1]
            heapq.heappush(tasks, (sim_r, cand))
            if len(tasks) > 10:
                heapq.heappop(tasks)
        file.write(word + '\t')
        file.write(' '.join([str(task[1]) for task in tasks]))
        file.write('\n')
        file.flush()
file.close()