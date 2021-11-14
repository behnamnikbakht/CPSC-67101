import text2emotion as te

from time import time

dataset = open('dataset', 'r')

category_norm = {
    'anger': 'Angry',
    'fear': 'Fear',
    'joy': 'Happy',
    'sadness': 'Sad',
    'surprise': 'Surprise',
}

stat = {
    'matched': 0,
    'total': 0,
}

t1 = time()
i = 0

for line in dataset.readlines():
    s = line.strip().split(";")
    text = s[0]
    category = s[1]
    if not category in category_norm:
        continue
    category = category_norm[category]
    #print("text = {} , category = {}".format(text, category))
    emotion = te.get_emotion(text)
    max_confidence = max(emotion, key=emotion.get)
    if max_confidence == category:
        stat['matched'] = stat['matched'] + 1
    stat['total'] = stat['total'] + 1
    #print("m = {}, {}, {}".format(max_confidence, emotion[max_confidence], category))
    if i % 100 == 0:
        print("i = {}".format(i))
    i = i + 1

t2 = time()

print("stat = {}, total time = {}".format(stat, int(1000 * (t2 - t1))))

"""
result:
stat = {'matched': 5425, 'total': 14696}, total time = 1000165
"""