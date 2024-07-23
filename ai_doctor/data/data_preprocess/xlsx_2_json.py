import json, yaml, os, glob, uuid
import pandas as pd
from config.map_dictor_to_word import W_Mapping, TYPE_INFO, TYPE_INDC
from config.map_dictor_to_description import D_Mapping

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
        sh_df = sh_df.sample(frac=1).reset_index(drop=True)  # shuffle
        sh_df_label = sh_df.iloc[:, 0]
        sh_df_data = sh_df.iloc[:, 1:]
        file_uid = sh_name.split('.')[1] + '_Mapping' if len(sh_name.split('.')) > 1 else sh_name
        mapping = W_Mapping[file_uid]
        if mapping["type"] == TYPE_INFO:
            for k, v in mapping["rule"].items():
                sh_df_data[k] = sh_df_data[k].replace(v)
        elif mapping["type"] == TYPE_INDC:
            for k, v in mapping["rule"].items():
                if not v: continue  # continue if indicator doesn't have rules
                sh_df_data[k] = pd.cut(sh_df_data[k], bins=v["bins"], labels=v["labels"], right=False,
                                       include_lowest=False)
        else:
            print("ERROR: this type is not defined......")
        sh_df_data.columns = [abbr_map.get(col, col) if not pd.isna(abbr_map.get(col)) else col for col in
                              sh_df_data.columns]

        # # save to file
        # sh_df.to_json((save_word_json_path + file_names['word_data_json']).format(sh_name.replace('.', '_')),
        #               orient='records', force_ascii=False)
        patients_info.append(json.loads(sh_df_data.to_json(orient='records', force_ascii=False)))
        labels_info.append(sh_df_label.values.tolist())
    return patients_info, labels_info


def process_row(row, rule_map, abbr_map):
    row_data = []
    # if rule_map['type'] == TYPE_INFO:
    #     return row.values
    for col in row.index:
        col_full_name = abbr_map.get(col, col) if not pd.isna(abbr_map.get(col)) else col
        if col_full_name == "label": continue
        if rule_map['type'] == TYPE_INFO and col in rule_map['rule']:
            row_data.append(row[col])
        else:
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
        if rule_map["type"] == TYPE_INFO:
            for k, v in rule_map["rule"].items():
                sh_df[k] = sh_df[k].replace(v)
        elif rule_map["type"] == TYPE_INDC:
            for k, v in rule_map["rule"].items():
                if not v: continue
                sh_df[k] = pd.cut(sh_df[k], bins=v["bins"], labels=v["labels"], right=False, include_lowest=False)
        else:
            print("ERROR: this type is not defined......")
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
        user_value = s1_conf['prompt']['finetune_align_prefix'] + str(patients_word_infos[0][i])
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


def create_train_test_dataset_diagnose():
    patients_word_infos, labels_infos = replace_indicator_to_word()
    patient_cnt = min([len(p) for p in patients_word_infos])
    train_dataset, test_dataset = [], []
    ratio = patient_cnt * fine_tune_diagnose_conf['train_data_ratio']
    for i in range(int(ratio)):
        user_value = (s1_conf['prompt']['finetune_align_prefix']
                      + str(patients_word_infos[0][i])
                      + "。"
                      + s1_conf['prompt']['finetune_diagnose_require'])
        ass_value = "诊断结果为：圆锥角膜病。" if labels_infos[0][i] else "诊断结果为：角膜正常。"
        patient_description = {'id': str(uuid.uuid4()),
                               'conversations': [{'from': 'user', 'value': user_value},
                                                 {'from': 'assistant', 'value': ass_value}],
                               'type': 'stage3'}

        train_dataset.append(patient_description)
    for i in range(int(ratio), patient_cnt):
        patient_description = (s1_conf['prompt']['finetune_align_prefix']
                               + str(patients_word_infos[0][i])
                               + "。"
                               # + s1_conf['prompt']['finetune_diagnose_require']
                               # + s1_conf['prompt']['diagnose_in_context_learning']
                               # + s1_conf['prompt']['diagnose_prompt_ltsbs']
                               # + s1_conf['prompt']['diagnose_prompt_tools']
                               )
        test_dataset.append(patient_description)
    return train_dataset, test_dataset, labels_infos[0][int(ratio):]


def create_test_dataset_diagnose():
    patients_word_infos, labels_infos = replace_indicator_to_word()
    patient_cnt = min([len(p) for p in patients_word_infos])
    test_dataset = []
    ratio = patient_cnt * fine_tune_diagnose_conf['train_data_ratio']
    # print("前：", labels_infos[0][:int(ratio)])
    # print("后：", labels_infos[0][int(ratio):])
    for i in range(int(ratio), patient_cnt):
        patient_description = (s1_conf['prompt']['finetune_align_prefix']
                               + str(patients_word_infos[0][i])
                               + "。"
                               # + s1_conf['prompt']['finetune_diagnose_require']
                               # + s1_conf['prompt']['diagnose_in_context_learning']
                               # + s1_conf['prompt']['diagnose_prompt_ltsbs']
                               # + s1_conf['prompt']['diagnose_prompt_tools']
                               )

        # ass_value = "诊断结果为：圆锥角膜病。" if labels_infos[0][i] else "诊断结果为：角膜正常。"

        test_dataset.append(patient_description)

    return test_dataset, labels_infos[0][int(ratio):]


def main():
    # trans_org_xlsx_to_json()  # transfer original data format from .xlsx to .json
    # replace_indicator_to_word()  # replace indicator from number to word
    # replace_indicator_to_description()  # replace indicator from number to description

    # # TODO: 1 create finetune-stage-1 dataset for aligning tabel with text
    # dataset = create_finetune_dataset_align()
    # with open(save_align_finetune_dataset_path + file_names['align_finetune_dataset_json'], 'w') as f:
    #     json.dump(dataset, f, ensure_ascii=False)

    # # TODO: 2 create finetune-stage-2 dataset for prediction
    train_dataset, test_dataset, label_info = create_train_test_dataset_diagnose()
    # with open(save_diagnose_finetune_dataset_path + file_names['diagnose_finetune_dataset_json'], 'w') as f:
    #     json.dump(train_dataset, f, ensure_ascii=False)
    # with open(save_diagnose_test_dataset_path + file_names['diagnose_test_dataset_json'], 'w') as f:
    #     json.dump(test_dataset, f, ensure_ascii=False)
    # with open(save_diagnose_test_label_path + file_names['diagnose_test_label_json'], 'w') as f:
    #     json.dump(label_info, f, ensure_ascii=False)
    # test_dataset, labels_info = create_test_dataset_diagnose()

if __name__ == '__main__':
    main()
