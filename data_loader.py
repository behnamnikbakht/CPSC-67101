def __read__(path, text_index, cls_index, splitter, normalizer):
    result = []
    f = open(path, 'r')
    for line in f.readlines():
        s = line.strip().split(splitter)
        text = s[text_index]
        cls = s[cls_index]
        text, cls = normalizer(text, cls)
        if cls is not None:
            result.append((text, cls))
    return result


class DataLoader:

    def __init__(self):
        pass

    def load1(self):
        def normalizer(text, cls):
            return text, cls.lower()
        dataset = __read__('data/labeled_dataset.txt', 0, 1, ";", normalizer)
        return dataset

    def load2(self):
        def normalizer(text, cls):
            cls = cls.replace('"', '').lower()
            if cls != "empty":
                return text.replace('"', ''), cls
            return text, None
        dataset = __read__('data/labeled_dataset2.csv', 3, 1, ",", normalizer)
        return dataset

