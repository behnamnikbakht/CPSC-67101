import text2emotion as te

from time import time

dataset = open('dataset2', 'r')

'worry', '', '', '', '', '', '', '', ''

category_norm = {
    'anger': 'Angry',
    'hate': 'Angry',
    'worry': 'Fear',
    'enthusiasm': 'Happy',
    'happiness': 'Happy',
    'love': 'Happy',
    'fun': 'Happy',
    'relief': 'Happy',
    'sadness': 'Sad',
    'surprise': 'Surprise',
}

stat = {
    'matched': 0,
    'total': 0,
}

t1 = time()
i = 0

result = {}

for line in dataset.readlines():
    s = line.strip().split(",")
    category = s[1].replace('"', '')
    text = s[3]
    #if not category in category_norm:
    #    continue
    #print("text = {} , category = {}".format(text, category))
    emotion = te.get_emotion(text)
    max_confidence = max(emotion, key=emotion.get)
    if max_confidence == category:
        stat['matched'] = stat['matched'] + 1
    stat['total'] = stat['total'] + 1
    if i % 100 == 0:
        print("i = {}".format(i))
    if max_confidence in result:
        c = result[max_confidence]
    else:
        c = {}
    if category in c:
        c[category] = c[category] + 1
    else:
        c[category] = 1
    result[max_confidence] = c
    i = i + 1

t2 = time()

print("result = {}".format(result))

print("stat = {}, total time = {}".format(stat, int(1000 * (t2 - t1))))

"""
result:
stat = {'matched': 5425, 'total': 14696}, total time = 1000165
"""