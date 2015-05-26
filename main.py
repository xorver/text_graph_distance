import re
import collections
import math
from scipy.sparse import dok_matrix
import scipy.sparse

ignored_chars = {'$', '(', ',', '.', ':', ';', '0', '3', '3', '3', '4', '5',
                 '6', '7', '8', '9', '\\', '`', '\'', '+', '-', '*', '/', '<',
                 '>', '^', '%', '=', '?', '!', '[', ']', '{', '}', '_', '\n',
                 '"', '&', '~'}

stop_words = {'a', 'aby', 'ach', 'acz', 'aczkolwiek', 'aj', 'albo', 'ale',
              'ależ', 'ani', 'aż', 'bardziej', 'bardzo', 'bo', 'bowiem', 'by',
              'byli', 'bynajmniej', 'być', 'był', 'była', 'było', 'były',
              'będzie', 'będą', 'cali', 'cała', 'cały', 'ci', 'cię', 'ciebie',
              'co', 'cokolwiek', 'coś', 'czasami', 'czasem', 'czemu', 'czy',
              'czyli', 'daleko', 'dla', 'dlaczego', 'dlatego', 'do', 'dobrze',
              'dokąd', 'dość', 'dużo', 'dwa', 'dwaj', 'dwie', 'dwoje', 'dziś',
              'dzisiaj', 'gdy', 'gdyby', 'gdyż', 'gdzie', 'gdziekolwiek',
              'gdzieś', 'i', 'ich', 'ile', 'im', 'inna', 'inne', 'inny',
              'innych', 'iż', 'ja', 'ją', 'jak', 'jakaś', 'jakby', 'jaki',
              'jakichś', 'jakie', 'jakiś', 'jakiż', 'jakkolwiek', 'jako',
              'jakoś', 'je', 'jeden', 'jedna', 'jedno', 'jednak', 'jednakże',
              'jego', 'jej', 'jemu', 'jest', 'jestem', 'jeszcze', 'jeśli',
              'jeżeli', 'już', 'ją', 'każdy', 'kiedy', 'kilka', 'kimś', 'kto',
              'ktokolwiek', 'ktoś', 'która', 'które', 'którego', 'której',
              'który', 'których', 'którym', 'którzy', 'ku', 'lat', 'lecz',
              'lub', 'ma', 'mają', 'mało', 'mam', 'mi', 'mimo', 'między',
              'mną', 'mnie', 'mogą', 'moi', 'moim', 'moja', 'moje', 'może',
              'możliwe', 'można', 'mój', 'mu', 'musi', 'my', 'na', 'nad',
              'nam', 'nami', 'nas', 'nasi', 'nasz', 'nasza', 'nasze',
              'naszego', 'naszych', 'natomiast', 'natychmiast', 'nawet',
              'nią', 'nic', 'nich', 'nie', 'niech', 'niego', 'niej', 'niemu',
              'nigdy', 'nim', 'nimi', 'niż', 'no', 'o', 'obok', 'od', 'około',
              'on', 'ona', 'one', 'oni', 'ono', 'oraz', 'oto', 'owszem', 'pan',
              'pana', 'pani', 'po', 'pod', 'podczas', 'pomimo', 'ponad',
              'ponieważ', 'powinien', 'powinna', 'powinni', 'powinno', 'poza',
              'prawie', 'przecież', 'przed', 'przede', 'przedtem', 'przez',
              'przy', 'roku', 'również', 'sama', 'są', 'się', 'skąd', 'sobie',
              'sobą', 'sposób', 'swoje', 'ta', 'tak', 'taka', 'taki', 'takie',
              'także', 'tam', 'te', 'tego', 'tej', 'temu', 'ten', 'teraz',
              'też', 'to', 'tobą', 'tobie', 'toteż', 'trzeba', 'tu', 'tutaj',
              'twoi', 'twoim', 'twoja', 'twoje', 'twym', 'twój', 'ty', 'tych',
              'tylko', 'tym', 'u', 'w', 'wam', 'wami', 'was', 'wasz', 'wasza',
              'wasze', 'we', 'według', 'wiele', 'wielu', 'więc', 'więcej',
              'wszyscy', 'wszystkich', 'wszystkie', 'wszystkim', 'wszystko',
              'wtedy', 'wy', 'właśnie', 'z', 'za', 'zapewne', 'zawsze', 'ze',
              'zł', 'znowu', 'znów', 'został', 'żaden', 'żadna', 'żadne',
              'żadnych', 'że', 'żeby'}


def base_forms(line):
    tokens = line.lower()[:-1].split(", ")
    return tokens


def normalize_text(text):
    text = text.lower()
    for pattern in ignored_chars:
        text = re.sub(re.escape(pattern), '', text)
    return text.split()


def to_base(args):
    (word_list, base_form) = args
    counter = collections.Counter()
    for word in word_list:
        try:
            counter[base_form[word]] += 1
        except KeyError:
            counter[word] += 1
    return counter


def norm(x):
    return max(0.0001, math.sqrt(dot_product(x, x)))


def dot_product(x, y):
    [i1, j1, _] = scipy.sparse.find(x)
    [i2, j2, _] = scipy.sparse.find(y)
    set1 = set(zip(i1, j1))
    set2 = set(zip(i2, j2))
    return sum([x[i, j] * y[i, j] for (i, j) in set1 & set2])


def cosine(x, y):
    return 1 - (1.0 * dot_product(x, y)) / (norm(x) * norm(y))


def create_graph(notice, corpus_index, k):
    graph = dok_matrix((len(corpus_index)+1, len(corpus_index)+1))
    for i in range(len(notice)):
        for word in notice[i:i+k+1]:
            try:
                graph[corpus_index[notice[i]], corpus_index[word]] += 1
            except IndexError:
                pass
            except KeyError:
                pass
    return graph


# read odm
base_form = {}
corpus_words = []
with open("data/odm_utf8.txt") as file:
    for line in file.readlines():
        tokens = base_forms(line)
        corpus_words.append(tokens[0])
        for el in tokens:
            base_form[el] = tokens[0]
corpus_words = [x for x in corpus_words if x not in stop_words]
corpus_index = {}
for i in range(len(corpus_words)):
    corpus_index[corpus_words[i]] = i

# read pap
with open("data/pap.txt") as file:
    text = file.read()
notice_text = re.split(r'#.*', text)[1:10000]
notice_words = [normalize_text(x) for x in notice_text]
notice_words = [[base_form.get(word, '') for word in words] for words in notice_words]
notice_words = [[word for word in words if word != ''] for words in notice_words]
notice_words = [[word for word in words if word not in stop_words] for words in notice_words]

for k in range(5):
    print("================================")
    print(k)
    print("================================")
    for i in range(10, 20):
        print("--------------------------------")
        print(notice_text[i])
        print("--------------------------------")
        notice_graphs = [create_graph(notice, corpus_index, k) for notice in notice_words]
        index_to_distance = [(j, cosine(notice_graphs[j], notice_graphs[i])) for j in range(len(notice_text))]
        best_notices = sorted(index_to_distance, key=lambda item: item[1])
        for (best_index, _) in best_notices[:10]:
            print("- - - - - - - - - - - - - - - - ")
            print(notice_text[best_index])