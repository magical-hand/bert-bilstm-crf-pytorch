# coding=utf-8

def read_tsv(file_path, split_seg="\t"):
    """
    read tsv style data1
    :param file_path: file path
    :param split_seg: seg
    :return: [(sentence, label), ...]
    """
    data = []
    sentence = []
    label = []
    with open(file_path, 'r', encoding='utf8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                if sentence:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            if split_seg not in line:
                split_seg = " "
            splits = line.split(split_seg)
            sentence.append(splits[0])
            label.append(splits[-1])
    if sentence:
        data.append((sentence, label))
    return data
Note=open("data/NER/People's Daily/dev.txt", mode='w+', encoding='utf8')
write_data=read_tsv("data/NER/People's Daily/example.dev")
for data1 in write_data:
    for data2 in data1[0]:
        Note.write(data2+' ')
    Note.write('|||')
    for data3 in data1[1]:
        Note.write(data3+' ')
    Note.write('\n')
Note.close()
