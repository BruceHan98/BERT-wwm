import pickle
from tqdm import tqdm
import jieba


def print_train_sample(train_data_file):
    train_file = open(train_data_file, 'rb')
    train_data = pickle.load(train_file)
    train_file.close()

    print(f"Train samples num: {len(train_data)}")
    print("Train sample 0:")
    print(train_data[0])


def print_dev_sample(dev_data_file, dev_gt_file):
    dev_file = open(dev_data_file, 'rb')
    dev_data = pickle.load(dev_file)
    dev_file.close()

    dev_gt_file = open(dev_gt_file, 'rb')
    dev_gt = pickle.load(dev_gt_file)
    dev_gt_file.close()

    print(f"Dev samples num: {len(dev_data)}")
    print("Dev sample 0:")
    print(dev_data[0])

    print(f"Dev gt num: {len(dev_gt)}")
    print("Dev gt 0:")
    print(dev_gt[0])


def print_test_sample(test_data_file, test_gt_file):
    test_file = open(test_data_file, 'rb')
    test_data = pickle.load(test_file)
    test_file.close()

    test_gt_file = open(test_gt_file, 'rb')
    test_gt = pickle.load(test_gt_file)
    test_gt_file.close()

    print(f"Test samples num: {len(test_data)}")
    print("Test sample 0:")
    print(test_data[0])

    print(f"Test gt num: {len(test_gt)}")
    print("Test gt 0:")
    print(test_gt[0])


def split_long_sentence(long_sentence, max_length):
    split_sentences = []
    end = min(len(long_sentence), max_length)
    split_idx = long_sentence[:end][::-1].find('。')
    if split_idx == -1:
        split_idx = long_sentence[:end][::-1].find('，')
        if split_idx == -1:
            split_idx = long_sentence[:end][::-1].find('！')
    if split_idx == -1:
        return [long_sentence]

    split_idx = end - split_idx
    split_sentences.append(long_sentence[:split_idx])

    if len(long_sentence[split_idx:]) < max_length:
        split_sentences.append(long_sentence[split_idx:])
        return split_sentences
    else:
        split_sentences.extend(split_long_sentence(long_sentence[split_idx:], max_length))
        return split_sentences


def create_train_data(train_data_file, len_limit=256):
    train_file = open(train_data_file, 'rb')
    train_data = pickle.load(train_file)
    train_file.close()

    train_dialog_raw = open('../data/mdg/train_dialog_raw.txt', 'w', encoding='utf-8')
    train_dialog_merge = open('../data/mdg/train_dialog_merge.txt', 'w', encoding='utf-8')
    train_dialog_info = open('../data/mdg/train_dialog_info.txt', 'w', encoding='utf-8')

    all_max_len, all_min_len, all_turn = 0, 999, 0
    out_of_len_num = 0
    used_sample_num = 0
    for i, dialog in enumerate(tqdm(train_data)):  # data: list
        last_id = ''
        buffer = ''
        conversations = []
        raw_conversations = []
        used_sample = True
        turn = 0
        for j, utter in enumerate(dialog):  # utter: dict
            id_ = utter['id']  # string
            sentence = utter['Sentence']  # string

            if len(sentence) > len_limit:
                if last_id != '' and id_ != last_id:
                    turn += 1
                split_sentence_list = split_long_sentence(sentence, len_limit)
                if len(split_sentence_list) == 1:
                    out_of_len_num += 1
                    used_sample = False
                    break
                # print(split_sentence_list)
                if buffer != '':
                    conversations.append((last_id, buffer))
                for k, sent in enumerate(split_sentence_list):
                    if len(sent) > len_limit:
                        out_of_len_num += 1
                        used_sample = False
                        break
                    raw_conversations.append(sent)
                    if k != len(split_sentence_list) - 1:
                        conversations.append((id_, sent))
                    else:
                        buffer = sent
                last_id = id_
                continue
            # 纯对话和实体
            raw_conversations.append(sentence)
            # 融合相同角色对话
            if j == 0:
                last_id = id_
                buffer += sentence
            elif j != len(dialog) - 1:
                if id_ == last_id:
                    if len(buffer) + len(sentence) > len_limit:
                        conversations.append((last_id, buffer))
                        buffer = sentence
                    else:
                        buffer += sentence
                else:
                    turn += 1
                    conversations.append((last_id, buffer))
                    buffer = sentence
                    last_id = id_
            else:
                if id_ == last_id:
                    if last_id == "Patient":
                        break
                    else:
                        turn += 1
                        if len(buffer) + len(sentence) > len_limit:
                            conversations.append((last_id, buffer))
                            conversations.append((last_id, sentence))
                        else:
                            conversations.append((last_id, buffer + sentence))
                else:
                    if last_id == "Patient":
                        turn += 1
                        conversations.append((last_id, buffer))
                        conversations.append((id_, sentence))
                    else:
                        break
        if used_sample:
            used_sample_num += 1
            for sentence in raw_conversations:
                train_dialog_raw.write(sentence + '\n')
            train_dialog_raw.write('\n')
            max_len, min_len = 0, 999
            for (id_, sentence) in conversations:
                if len(sentence) < 5:
                    continue
                max_len = max(max_len, len(sentence))
                # if len(sentence) > len_limit:
                #     print(len(sentence), id_, sentence)
                min_len = min(min_len, len(sentence))
                train_dialog_merge.write(sentence + '\n')
            turn = turn // 2
            all_turn += turn
            all_max_len = max(all_max_len, max_len)
            all_min_len = min(all_min_len, min_len)
            train_dialog_merge.write('\n')
            train_dialog_info.write(f"{max_len}\t{min_len}\t{turn}\n")

    train_dialog_raw.close()
    train_dialog_merge.close()
    train_dialog_info.close()
    print(f"Max sentence len: {all_max_len}")
    print(f"Min sentence len: {all_min_len}")
    print(f"Average turns: {all_turn / used_sample_num:.2f}")
    print(f"Out of len num: {out_of_len_num}, {out_of_len_num / len(train_data) * 100:.2f}%")


def create_dev_data(dev_data_file, dev_gt_file, len_limit=256):
    dev_file = open(dev_data_file, 'rb')
    dev_gt_file = open(dev_gt_file, 'rb')
    dev_data = pickle.load(dev_file)
    dev_gt = pickle.load(dev_gt_file)
    dev_file.close()
    dev_gt_file.close()

    dev_dialog_raw = open('../data/mdg/dev_dialog_raw.txt', 'w', encoding='utf-8')
    dev_dialog_merge = open('../data/mdg/dev_dialog_merge.txt', 'w', encoding='utf-8')
    dev_dialog_info = open('../data/mdg/dev_dialog_info.txt', 'w', encoding='utf-8')

    all_max_len, all_min_len, all_turn = 0, 999, 0
    out_of_len_num = 0
    used_sample_num = 0
    for i, sample in enumerate(dev_data):
        last_id = ''
        buffer = ''
        conversations = []
        raw_conversations = []
        used_sample = True
        dialog = sample['history']
        turn = 0
        for j, utter in enumerate(dialog):
            id_ = utter[:2]
            sentence = utter[3:]

            if len(sentence) > len_limit:
                if last_id != '' and id_ != last_id:
                    turn += 1
                split_sentence_list = split_long_sentence(sentence, len_limit)
                if len(split_sentence_list) == 1:
                    out_of_len_num += 1
                    used_sample = False
                    break
                # print(split_sentence_list)
                if buffer != '':
                    conversations.append((last_id, buffer))
                for k, sent in enumerate(split_sentence_list):
                    if len(sent) > len_limit:
                        out_of_len_num += 1
                        used_sample = False
                        break
                    raw_conversations.append(sent)
                    if k != len(split_sentence_list) - 1:
                        conversations.append((id_, sent))
                    else:
                        buffer = sent
                last_id = id_
                continue

            # 纯对话和实体
            raw_conversations.append(sentence)
            # 融合相同角色对话
            if j == 0:
                last_id = id_
                buffer += sentence
                continue
            elif j != len(dialog) - 1:
                if id_ == last_id:
                    if len(buffer) + len(sentence) > len_limit:
                        conversations.append((last_id, buffer))
                        buffer = sentence
                    else:
                        buffer += sentence
                else:
                    turn += 1
                    conversations.append((last_id, buffer))
                    buffer = sentence
                    last_id = id_
            else:
                if id_ == last_id:
                    if len(buffer) + len(sentence) > len_limit:
                        conversations.append((last_id, buffer))
                        conversations.append((last_id, sentence))
                    else:
                        buffer += sentence
                        conversations.append((last_id, buffer))
                else:
                    turn += 1
                    conversations.append((last_id, buffer))
                    conversations.append((id_, sentence))
        if used_sample:
            used_sample_num += 1
            for sentence in raw_conversations:
                dev_dialog_raw.write(sentence + '\n')
            dev_dialog_raw.write('\n')
            max_len, min_len = 0, 999
            for (id_, sentence) in conversations:
                if len(sentence) < 5:
                    continue
                max_len = max(max_len, len(sentence))
                min_len = min(min_len, len(sentence))
                dev_dialog_merge.write(sentence + '\n')
            turn = turn / 2
            all_turn += turn
            all_max_len = max(all_max_len, max_len)
            all_min_len = min(all_min_len, min_len)
            dev_dialog_merge.write(dev_gt[i] + '\n')
            dev_dialog_merge.write('\n')
            dev_dialog_info.write(f"{max_len}\t{min_len}\t{turn}\n")
    dev_dialog_raw.close()
    dev_dialog_merge.close()
    dev_dialog_info.close()
    print(f"Max sentence len: {all_max_len}")
    print(f"Min sentence len: {all_min_len}")
    print(f"Average turns: {all_turn / used_sample_num:.2f}")
    print(f"Out of len num: {out_of_len_num}, {out_of_len_num / len(dev_data) * 100:.2f}%")


def do_segment(input_file, output_file):
    jieba.load_userdict('../data/mdg/entity_list.txt')
    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin.readlines()):
            line = line.strip()
            if line:
                segment = jieba.lcut(line)
                fout.write(' '.join(segment) + '\n')
            else:
                fout.write('\n')


if __name__ == '__main__':
    train_file_path = '../data/mdg/train.pk'
    dev_file_path = '../data/mdg/dev.pk'
    dev_gt_path = '../data/mdg/dev_gt.pk'
    test_file_path = '../data/mdg/test.pk'
    test_gt_path = '../data/mdg/test_gt.pk'

    # print_train_sample(train_file_path)
    # print_dev_sample(dev_file_path, dev_gt_path)
    # print_test_sample(test_file_path, test_gt_path)
    # create_train_data(train_file_path)
    # create_dev_data(dev_file_path, dev_gt_path)
    do_segment('../data/mdg/train_dialog_merge.txt', '../data/mdg/train_dialog_seg.txt')
    do_segment('../data/mdg/dev_dialog_merge.txt', '../data/mdg/dev_dialog_seg.txt')
