import json, yaml, os, glob, uuid
import random

import pandas as pd
import numpy as np
from math import gcd
from functools import reduce
import logging
from config.map_dictor_to_word import W_Mapping, TYPE_INFO, TYPE_INDC
from config.map_dictor_to_description import D_Mapping
import argparse

# Language
LANGUAGE = 'en'
TASK = 'MC'  # 'SC'

# 1 load config
conf_path = r'/config/s1_align_conf.yaml'
with open(os.path.dirname(__file__) + conf_path, 'r') as s1_yaml:
    s1_conf = yaml.safe_load(s1_yaml)
file_pathes = s1_conf['file_path']
file_names = s1_conf['file_name']
load_path = file_pathes['load']
save_org_json_path = file_pathes['save_org']
save_word_json_path = file_pathes['save_word']
save_align_finetune_dataset_path = file_pathes['save_align_finetune_dataset']
save_diagnose_finetune_dataset_path = file_pathes['save_diagnose_finetune_dataset']
save_diagnose_test_dataset_path = file_pathes['save_diagnose_test_dataset']
save_diagnose_test_label_path = file_pathes['save_diagnose_test_label']
fine_tune_diagnose_conf = s1_conf['fine_tune']['diagnose']

prompt_conf = s1_conf['prompt'][LANGUAGE]

table_ids = s1_conf['table_ids']  # table ids
label_ids = s1_conf['label_ids']  # lable ids

train_data_ratio = fine_tune_diagnose_conf['train_data_ratio']
test_data_ratio = fine_tune_diagnose_conf['test_data_ratio']

logging.basicConfig(format="[%(asctime)s]: %(message)s", datefmt="%Y-%M-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger('-')


def trans_org_xlsx_to_json():
    am_df = pd.read_excel(load_path + file_names['abbr_mapping'], sheet_name=0)
    abbr_map = dict(zip(am_df[am_df.columns[0]], am_df[am_df.columns[1]]))
    # print(abbr_map)
    df = pd.read_excel(load_path + file_names['org_data_xlsx'], sheet_name=None)
    sheet_names = list(df.keys())
    print(sheet_names)

    for sh_name in sheet_names:
        sh_df = df[sh_name]
        # print(sh_df.columns)
        sh_df.columns = [abbr_map.get(col, col) if not pd.isna(abbr_map.get(col)) else col for col in sh_df.columns]
        # print(sh_df.columns)
        # file_uid = sh_name.split('.')[1] if len(sh_name.split('.')) > 1 else sh_name
        # print((save_path + file_names['org_data_json']).format(sh_name.replace('.', '_')))
        sh_df.to_json((save_org_json_path + file_names['org_data_json']).format(sh_name.replace('.', '_')),
                      orient='records', force_ascii=False)


def replace_indicator_to_word():
    am_df = pd.read_excel(load_path + file_names['abbr_mapping'], sheet_name=0)
    abbr_map = dict(zip(am_df[am_df.columns[0]], am_df[am_df.columns[1]]))
    df = pd.read_excel(load_path + file_names['org_data_xlsx'], sheet_name=None)
    sheet_names = list(df.keys())
    print(sheet_names)
    patients_info = []
    labels_info = []
    for sh_name in sheet_names:
        sh_df = df[sh_name]
        # sh_df = sh_df.sample(frac=1).reset_index(drop=True)  # shuffle
        sh_df_label = sh_df.iloc[:, 0]
        sh_df_data = sh_df.iloc[:, 1:]
        file_uid = sh_name.split('.')[1] + '_Mapping' if len(sh_name.split('.')) > 1 else sh_name
        mapping = W_Mapping[file_uid]
        if mapping["type"] == TYPE_INDC:
            for k, v in mapping["rule"].items():
                if not v: continue  # continue if indicator doesn't have rules
                sh_df_data[k] = pd.cut(sh_df_data[k], bins=v["bins"], labels=v["labels"], right=False,
                                       include_lowest=False)
        sh_df_data.columns = [abbr_map.get(col, col) if not pd.isna(abbr_map.get(col)) else col for col in
                              sh_df_data.columns]
        be = "：" if sh_name == '1.DangerousFactor' else " is "  # TODO: 需要优化，暂时第一张用中文，第二种用英文
        rows_data = sh_df_data.apply(
            lambda row: ','.join(f"{col_name}{be}{row[col_name]}" for col_name in sh_df_data.columns), axis=1)
        rows_str = rows_data.values.tolist()

        #     # # save to file
        #     # sh_df.to_json((save_word_json_path + file_names['word_data_json']).format(sh_name.replace('.', '_')),
        #     #               orient='records', force_ascii=False)
        patients_info.append(rows_str)
        labels_info.append(sh_df_label.values.tolist())
    return patients_info, labels_info


def process_row(row, abbr_map):
    # row data: transfer json to text
    row_data = []
    for col in row.index:
        col_full_name = abbr_map.get(col, col) if not pd.isna(abbr_map.get(col)) else col
        if col_full_name == "label": continue
        row_data.append(f"{col_full_name}为{row[col]}")
    return ','.join(row_data) + '。'


def replace_indicator_to_description():
    am_df = pd.read_excel(load_path + file_names['abbr_mapping'], sheet_name=0)
    abbr_map = dict(zip(am_df[am_df.columns[0]], am_df[am_df.columns[1]]))
    df = pd.read_excel(load_path + file_names['org_data_xlsx'], sheet_name=None)
    sheet_names = list(df.keys())
    print(sheet_names)
    patients_info = []
    for sh_name in sheet_names:
        sh_df = df[sh_name]
        file_uid = sh_name.split('.')[1] + '_Mapping' if len(sh_name.split('.')) > 1 else sh_name
        rule_map = D_Mapping[file_uid]
        if rule_map["type"] == TYPE_INDC:
            for k, v in rule_map["rule"].items():
                if not v: continue
                sh_df[k] = pd.cut(sh_df[k], bins=v["bins"], labels=v["labels"], right=False, include_lowest=False)
        patients_disease_description = sh_df.apply(lambda row: process_row(row, rule_map, abbr_map), axis=1)
        patients_info.append(patients_disease_description.values)

    return patients_info
    # patient_cnt = min([len(p) for p in patients_info])
    #
    # dataset = []
    # for i in range(patient_cnt):
    #     patient_description = {'id': uuid.uuid4(),
    #                            'conversations': [{'from': 'user', 'value': ''}, {'from': 'assistant', 'value': ''}]}
    #     user_value =
    # sh_df.columns = [abbr_map.get(col, col) if not pd.isna(abbr_map.get(col)) else col for col in sh_df.columns]
    # sh_df.to_json((save_word_json_path + file_names['word_data_json']).format(sh_name.replace('.', '_')),
    #               orient='records', force_ascii=False)


def create_finetune_dataset_align():
    patients_word_infos, _ = replace_indicator_to_word()
    patients_description_infos = replace_indicator_to_description()
    patient_cnt = min([len(p) for p in patients_word_infos])
    dataset = []
    for i in range(patient_cnt):
        user_value = prompt_conf['finetune_align_prefix'] + str(patients_word_infos[0][i])
        user_description = patients_description_infos[0][i]
        patient_description = {'id': str(uuid.uuid4()),
                               'conversations': [{'from': 'user', 'value': user_value},
                                                 {'from': 'assistant', 'value': user_description}],
                               'type': 'stage2'}

        # print(user_value)
        # print(user_description)
        # print(patient_description)
        dataset.append(patient_description)

    return dataset


def generate_patients_word_infos(patients_word_infos, labels_infos):
    # for tid in table_ids:
    pswis, lbis = np.array(patients_word_infos).T.tolist(), np.array(
        labels_infos).T.tolist()  # [[p_1_abc],[p_2_def],[p_3_ghi]]->[[p_1_adg],[]]
    patients_word_infos = [','.join(p_infos[i] for i in table_ids) for p_infos in pswis]
    # label_infos = [1 if sum(labels) else 0 for labels in lbis]
    label_infos = [labels[0] for labels in lbis]
    return patients_word_infos, label_infos


def split_dataset(patients_word_infos, labels_infos):  # 拼接后的信息
    train_data_set, test_data_set, train_data_labels, test_data_labels = [], [], [], []
    dataset_size = len(labels_infos)
    lable_counts = [labels_infos.count(lid) for lid in label_ids]
    # non_zero_ratios_label = [num for num in ratios_label if num != 0]
    # gcd_ratios = reduce(gcd, non_zero_ratios_label)  # greatest common divisor
    # ratios_label_simple = [num // gcd_ratios for num in ratios_label]
    logger.info('----------- label count ----------')
    logger.info(lable_counts)
    logger.info('----------------------------------')
    sample_counts = [round(train_data_ratio * num) for num in lable_counts]
    logger.info('----------- sample count ----------')
    logger.info(sample_counts)
    logger.info('----------------------------------')
    cur_counts = [0] * len(label_ids)
    for i in range(dataset_size):
        lable = labels_infos[i]
        if cur_counts[lable] < sample_counts[lable]:
            train_data_set.append(patients_word_infos[i])
            train_data_labels.append(lable)
            cur_counts[lable] += 1
        else:
            test_data_set.append(patients_word_infos[i])
            test_data_labels.append(lable)
    logger.info('----------- split count ----------')
    logger.info(cur_counts)
    logger.info('----------------------------------')
    logger.info('----------- data set info ----------')
    logger.info('train_data_set: %d', len(train_data_set))
    logger.info('train_data_labels: %d', len(train_data_labels))
    logger.info('test_data_set: %d', len(test_data_set))
    logger.info('test_data_labels: %d', len(test_data_labels))
    logger.info('----------------------------------')
    return train_data_set, test_data_set, train_data_labels, test_data_labels


def shuffle_infos(patients_word_infos, labels_infos):
    paired_list = list(zip(patients_word_infos, labels_infos))
    random.shuffle(paired_list)
    patients_word_infos, labels_infos = zip(*paired_list)
    return list(patients_word_infos), list(labels_infos)


def generate_label(label, args):
    if args.task == "MC":
        if label == 1:
            return "forme fruste keratoconus"
        elif label == 2:
            return "subclinical keratoconus"
        elif label == 3:
            return "clinical keratoconus"
        else:
            return "No"
    else:
        return "Yes" if label else "No"


def create_train_test_dataset_diagnose(args):
    patients_word_infos, labels_infos = replace_indicator_to_word()
    patients_word_infos, labels_infos = generate_patients_word_infos(patients_word_infos, labels_infos)
    logger.info('----------- all label infos [count_0, count_1, count_2, count_3] ----------')
    logger.info('%d %d %d %d', labels_infos.count(0), labels_infos.count(1), labels_infos.count(2),
                labels_infos.count(3))
    logger.info('----------- ----------------------------------- ----------')
    patients_word_infos, labels_infos = shuffle_infos(patients_word_infos, labels_infos)
    org_train_dataset, org_test_dataset, train_data_labels, test_data_labels = split_dataset(patients_word_infos,
                                                                                             labels_infos)
    train_dataset, test_dataset = [], []
    for i, patient_info in enumerate(org_train_dataset):
        user_value = (prompt_conf['finetune_diagnose_prefix']
                      + patient_info
                      + "。"
                      + prompt_conf['finetune_diagnose_require']
                      + prompt_conf['diagnose_in_context_learning']
                      + prompt_conf['diagnose_prompt_ltsbs']
                      + prompt_conf['diagnose_prompt_tools']
                      )
        # ass_value = "诊断结果为：圆锥角膜病。" if labels_infos[0][i] else "诊断结果为：角膜正常。"
        ass_value = generate_label(train_data_labels[i], args)  # "Yes" if train_data_labels[i] else "No"
        patient_description = {'id': str(uuid.uuid4()),
                               'conversations': [{'from': 'user', 'value': user_value},
                                                 {'from': 'assistant', 'value': ass_value}],
                               'type': 'stage3'}

        train_dataset.append(patient_description)
    for i, patient_info in enumerate(org_test_dataset):
        patient_description = (prompt_conf['finetune_diagnose_prefix']
                               + patient_info
                               + "。"
                               + prompt_conf['finetune_diagnose_require']
                               + prompt_conf['diagnose_in_context_learning']
                               + prompt_conf['diagnose_prompt_ltsbs']
                               + prompt_conf['diagnose_prompt_tools']
                               )
        test_dataset.append(patient_description)
    return train_dataset, test_dataset, test_data_labels


def create_test_dataset_diagnose():
    patients_word_infos, labels_infos = replace_indicator_to_word()
    patient_cnt = min([len(p) for p in patients_word_infos])
    test_dataset = []
    ratio = patient_cnt * train_data_ratio
    # print("前：", labels_infos[0][:int(ratio)])
    # print("后：", labels_infos[0][int(ratio):])
    for i in range(int(ratio), patient_cnt):
        patient_description = (prompt_conf['finetune_align_prefix']
                               + str(patients_word_infos[0][i])
                               + "。"
                               + prompt_conf['finetune_diagnose_require']
                               + prompt_conf['diagnose_in_context_learning']
                               + prompt_conf['diagnose_prompt_ltsbs']
                               + prompt_conf['diagnose_prompt_tools']
                               )

        # ass_value = "诊断结果为：圆锥角膜病。" if labels_infos[0][i] else "诊断结果为：角膜正常。"

        test_dataset.append(patient_description)

    return test_dataset, labels_infos[0][int(ratio):]


def main(args):
    # trans_org_xlsx_to_json()  # transfer original data format from .xlsx to .json
    # patient_infos, labels = replace_indicator_to_word()  # replace indicator from number to word
    # replace_indicator_to_description()  # replace indicator from number to description

    # # TODO: 1 create finetune-stage-1 dataset for aligning tabel with text
    # dataset = create_finetune_dataset_align()
    # with open(save_align_finetune_dataset_path + file_names['align_finetune_dataset_json'], 'w') as f:
    #     json.dump(dataset, f, ensure_ascii=False)

    # test_dataset, labels_info = create_test_dataset_diagnose()

    # # # TODO: 2 create finetune-stage-2 dataset for prediction
    # create_train_test_dataset_diagnose()
    logger.info(f'table ids: {table_ids}')
    train_dataset, test_dataset, label_info = create_train_test_dataset_diagnose(args)
    # print(train_dataset[0], '\n', test_dataset[0], '\n', label_info[0])
    with open(save_diagnose_finetune_dataset_path + file_names['diagnose_finetune_dataset_json'], 'w') as f:
        json.dump(train_dataset, f, ensure_ascii=False)
    with open(save_diagnose_test_dataset_path + file_names['diagnose_test_dataset_json'], 'w') as f:
        json.dump(test_dataset, f, ensure_ascii=False)
    with open(save_diagnose_test_label_path + file_names['diagnose_test_label_json'], 'w') as f:
        json.dump(label_info, f, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test  checkpoint.")

    parser.add_argument(
        "-o", "--task", type=str, default="SC"
    )

    args = parser.parse_args()

    main(args)
