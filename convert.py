import os
import re

import datasets
from collections import OrderedDict

dirs = os.listdir("./KWDLC/knp")

# 下から2桁目が 0~1 のものを訓練用,
# 2 のものをテスト用とする.

train_dirs = [dir for dir in dirs if dir[-2] in ["0", "1"]]
valid_dirs = [dir for dir in dirs if dir[-2:] in ["20", "21"]]
test_dirs = [dir for dir in dirs if dir[-2:] in ["23", "24"]]

assert len(train_dirs) == 20
assert len(test_dirs) == 2
assert len(valid_dirs) == 2

ner_pat = re.compile(r'<NE:(.+):(.+)(:.+)*>')


train_data = []

for dir in train_dirs:
    files = os.listdir(f"./KWDLC/knp/{dir}")

    for file in files:
        with open(f"./KWDLC/knp/{dir}/{file}", "r") as f:
            cur_sent = ""
            cur_dict = []
            cur_ne = None
            cur_ne_type = None

            for line in f.readlines():
                split_line = line.split()

                if split_line[0] == "#":
                    # start new sentence
                    cur_sent = ""
                    cur_dict = []
                    cur_ne = None
                    cur_ne_type = None

                elif split_line[0] == "EOS":
                    train_data.append([tuple(dic) for dic in cur_dict])

                elif split_line[0] in ["*", "+"]:
                    # print(line)
                    res = ner_pat.search(line)
                    if res is not None:
                        cur_ne = res[2]
                        cur_ne_type = res[1][:3]

                else:
                    cur_sent += split_line[0]
                    cur_dict.append([split_line[0], "O"])

                    if cur_ne and cur_sent.endswith(cur_ne):
                        N = len(cur_dict)
                        tmp_word = ""

                        for i in range(N - 1, -1, -1):
                            cur_dict[i][1] = cur_ne_type
                            tmp_word = cur_dict[i][0] + tmp_word

                            if tmp_word == cur_ne:
                                break

                        cur_ne = None
                        cur_ne_type = None

print(f"#train data: {len(train_data)}")


valid_data = []

for dir in valid_dirs:
    files = os.listdir(f"./KWDLC/knp/{dir}")

    for file in files:
        with open(f"./KWDLC/knp/{dir}/{file}", "r") as f:
            cur_sent = ""
            cur_dict = []
            cur_ne = None
            cur_ne_type = None

            for line in f.readlines():
                split_line = line.split()

                if split_line[0] == "#":
                    # start new sentence
                    cur_sent = ""
                    cur_dict = []
                    cur_ne = None
                    cur_ne_type = None

                elif split_line[0] == "EOS":
                    valid_data.append([tuple(dic) for dic in cur_dict])

                elif split_line[0] in ["*", "+"]:
                    # print(line)
                    res = ner_pat.search(line)
                    if res is not None:
                        cur_ne = res[2]
                        cur_ne_type = res[1][:3]

                else:
                    cur_sent += split_line[0]
                    cur_dict.append([split_line[0], "O"])

                    if cur_ne and cur_sent.endswith(cur_ne):
                        N = len(cur_dict)
                        tmp_word = ""

                        for i in range(N - 1, -1, -1):
                            cur_dict[i][1] = cur_ne_type
                            tmp_word = cur_dict[i][0] + tmp_word

                            if tmp_word == cur_ne:
                                break

                        cur_ne = None
                        cur_ne_type = None

print(f"#train data: {len(valid_data)}")


test_data = []

for dir in test_dirs:
    files = os.listdir(f"./KWDLC/knp/{dir}")

    for file in files:
        with open(f"./KWDLC/knp/{dir}/{file}", "r") as f:
            cur_sent = ""
            cur_dict = []
            cur_ne = None
            cur_ne_type = None

            for line in f.readlines():
                split_line = line.split()

                if split_line[0] == "#":
                    # start new sentence
                    cur_sent = ""
                    cur_dict = []
                    cur_ne = None
                    cur_ne_type = None

                elif split_line[0] == "EOS":
                    test_data.append([tuple(dic) for dic in cur_dict])

                elif split_line[0] in ["*", "+"]:
                    # print(line)
                    res = ner_pat.search(line)
                    if res is not None:
                        cur_ne = res[2]
                        cur_ne_type = res[1][:3]

                else:
                    cur_sent += split_line[0]
                    cur_dict.append([split_line[0], "O"])

                    if cur_ne and cur_sent.endswith(cur_ne):
                        N = len(cur_dict)
                        tmp_word = ""

                        for i in range(N - 1, -1, -1):
                            cur_dict[i][1] = cur_ne_type
                            tmp_word = cur_dict[i][0] + tmp_word

                            if tmp_word == cur_ne:
                                break

                        cur_ne = None
                        cur_ne_type = None

print(f"#test data: {len(test_data)}")

def convert_to_dataset(data):
    entries = []
    all_labels = []
    for entry in data:
        text = []
        labels = []
        state = {'last_label': 'O', 'text_buffer': '', 'offset': 0}

        def flush():
            len_surface = len(state['text_buffer'])
            if state['last_label'] != 'O':
                labels.append({
                    'label': state['last_label'],
                    'start': state['offset'],
                    'end': state['offset'] + len_surface
                })
                all_labels.append(state['last_label'])
            text.append(state['text_buffer'])
            state['text_buffer'] = ''
            state['offset'] += len_surface

        for surface, label in entry:
            if state['last_label'] != label:
                flush()
            state['text_buffer'] += surface
            state['last_label'] = label
        flush()
        entries.append((''.join(text), labels))
    # レベルリストの重複を除く
    all_labels = sorted(set(all_labels))
    d = OrderedDict()
    d['text'] = [_[0] for _ in entries]
    d['tag_names'] = [all_labels] * len(entries)
    d['annotations'] = [_[1] for _ in entries]
    return datasets.Dataset.from_dict(d)

train_dataset = convert_to_dataset(train_data)
valid_dataset = convert_to_dataset(valid_data)
test_dataset = convert_to_dataset(test_data)
train_dataset.to_json('ner_train.json', force_ascii=False)
valid_dataset.to_json('ner_valid.json', force_ascii=False)
test_dataset.to_json('ner_test.json', force_ascii=False)
