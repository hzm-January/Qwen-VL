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
    for sh_name in sheet_names:
        sh_df = df[sh_name]
        file_uid = sh_name.split('.')[1] + '_Mapping' if len(sh_name.split('.')) > 1 else sh_name
        mapping = W_Mapping[file_uid]
        if mapping["type"] == TYPE_INFO:
            for k, v in mapping["rule"].items():
                sh_df[k] = sh_df[k].replace(v)
        elif mapping["type"] == TYPE_INDC:
            for k, v in mapping["rule"].items():
                if not v: continue
                sh_df[k] = pd.cut(sh_df[k], bins=v["bins"], labels=v["labels"], right=False, include_lowest=False)
        else:
            print("ERROR: this type is not defined......")
        sh_df.columns = [abbr_map.get(col, col) if not pd.isna(abbr_map.get(col)) else col for col in sh_df.columns]

        # # save to file
        # sh_df.to_json((save_word_json_path + file_names['word_data_json']).format(sh_name.replace('.', '_')),
        #               orient='records', force_ascii=False)
        patients_info.append(json.loads(sh_df.to_json(orient='records', force_ascii=False)))
    return patients_info


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


def create_finetune_dataset():
    patients_word_infos = replace_indicator_to_word()
    patients_description_infos = replace_indicator_to_description()
    patient_cnt = min([len(p) for p in patients_word_infos])
    dataset = []
    for i in range(patient_cnt):
        user_value = s1_conf['prompt']['finetune_align_prefix'] + str(patients_word_infos[0][i])
        user_description = patients_description_infos[0][i]
        patient_description = {'id': str(uuid.uuid4()),
                               'conversations': [{'from': 'user', 'value': user_value},
                                                 {'from': 'assistant', 'value': user_description}]}

        # print(user_value)
        # print(user_description)
        # print(patient_description)
        dataset.append(patient_description)

    return dataset


def main():
    # trans_org_xlsx_to_json()  # transfer original data format from .xlsx to .json
    # replace_indicator_to_word()  # replace indicator from number to word
    # replace_indicator_to_description()  # replace indicator from number to description
    dataset = create_finetune_dataset()
    with open(save_align_finetune_dataset_path + file_names['align_finetune_dataset_json'], 'w') as f:
        json.dump(dataset, f, ensure_ascii=False)


# 1 id

# 2 conversation

# col_names = sh_df.columns.values.tolist()
# print(col_names)
# print(type(col_names))
# for col_name in col_names:
#     print(col_name)


if __name__ == '__main__':
    main()

# 将数据和标题组合成字典
# print(3333, dict(zip(sh.row_values(0), sh.row_values(1))))
# 遍历excel，打印所有数据

save_f = 'text_info_json.jsonl'

# all_data = []
# for i in range(sh.nrows):
#     print(55555, sh.row_values(i))
#
#     # 组合为一段话
#     cur_data = {}
#     data = sh.row_values(i)
#
#     if i > 1:
#         cur_data['姓名'] = data[0]
#         cur_data['门诊时间'] = data[4]  # 后面匹配图像后删除
#         cur_data['性别'] = data[1]
#         cur_data['年龄'] = data[2]
#         cur_data['眼别'] = data[5]
#         print(type(data[3]))
#
#         cur_data['眼表疾病指数量表'] = str(round(data[6], 2)) if isinstance(data[6], float) else ''
#         # if cur_data['眼表疾病指数量表']:
#         # cur_data['眼表疾病指数量表']+='（眼表疾病指数量表：数值越大越趋向异常，数值越小越趋向正常，其中大于33表现为重度异常，大于23表现为中度异常，大于13表现为轻度异常，小于13表现为正常；）'
#         cur_data['角膜荧光染色评分'] = data[7] if isinstance(data[7], str) else str(data[7])
#         # if cur_data['角膜荧光染色评分'] :
#         #     cur_data['角膜荧光染色评分'] += '（角膜荧光染色评分：数值越大越趋向异常，数值越小越趋向正常，其中大于5表现为重度异常，大于3表现为中度异常，大于1表现为轻度异常，等于0表现为正常；）'
#         cur_data['泪膜破裂时间'] = data[8] if isinstance(data[8], str) else str(data[8])
#         # if cur_data['泪膜破裂时间']:
#         #     cur_data['泪膜破裂时间'] += '（泪膜破裂时间：数值越小越趋向异常，数值越大越趋向正常，其中小于2表现为重度异常，小于5表现为中度异常，小于10表现为轻度异常，大于10表现为正常；）'
#         cur_data['泪河高度'] = data[9] if isinstance(data[9], str) else str(data[9])
#         # if cur_data['泪河高度'] :
#         #     cur_data['泪河高度'] += '（泪河高度：数值越小越趋向异常，数值越大越趋向正常，其中小于0.05表现为重度异常，小于0.1表现为中度异常，小于0.2表现为轻度异常，大于0.2表现为正常；）'
#         # cur_data['视力'] = data[10] if isinstance(data[10], str) else str(data[10])
#         # cur_data['眼压'] = data[11] if isinstance(data[11], str) else str(data[11])
#         cur_data['泪液分泌实验'] = data[12] if isinstance(data[12], str) else str(data[12])
#         # if cur_data['泪液分泌实验']:
#         #     cur_data['泪液分泌实验'] += '（泪液分泌实验：数值越小越趋向异常，数值越大越趋向正常，其中小于10表现为异常，大于10表现为正常。）'
#
#         cur_data['您是否发生过皮肤排异'] = data[14]
#         cur_data['您是否发生过口腔排异'] = data[15]
#         cur_data['您是否发生过肠道排异'] = data[16]
#         cur_data['您是否发生过肺排异'] = data[17]
#         cur_data['您是否发生过肝排异'] = data[18]
#         cur_data['哭时，是否有眼泪'] = data[19]
#         cur_data['哭时有眼泪-流泪时感觉'] = data[20]
#         cur_data['哭时无眼泪-无泪时感觉'] = data[21]
#         cur_data['使用电子产品类型'] = data[22]
#         cur_data['每天平均电子产品使用时间'] = data[23] if isinstance(data[23], str) else str(data[23])
#
#         cnt = '在眼科诊断中，各个指标的严重程度分级为：\n眼表疾病指数量表：小于13正常，大于等于13且小于23轻度异常，大于等于23且小于33中度异常，大于等于33重度异常；\n右眼角膜荧光染色评分：0正常，1-2分轻度异常，3-4分中度异常，大于等于5分重度异常；\n右眼泪膜破裂时间：大于等于10s正常，6-10s轻度异常，2-5s中度异常，小于2s重度异常；\n右眼泪河高度：大于0.2正常，大于0.1且小于0.2轻度异常，大于0.05且小于等于0.1中度异常，小于等于0.05重度异常；\n右眼泪液分泌实验：大于10正常，小于等于10异常。'
#
#         cnt = '在眼科诊断中，各个指标的严重程度分级为：\n眼表疾病指数量表：数值越大越趋向异常，数值越小越趋向正常，其中大于33表现为重度异常，大于23表现为中度异常，大于13表现为轻度异常，小于13表现为正常；\n右眼角膜荧光染色评分：数值越大越趋向异常，数值越小越趋向正常，其中大于5表现为重度异常，大于3表现为中度异常，大于1表现为轻度异常，等于0表现为正常；\n右眼泪膜破裂时间：数值越小越趋向异常，数值越大越趋向正常，其中小于2表现为重度异常，小于5表现为中度异常，小于10表现为轻度异常，大于10表现为正常；\n右眼泪河高度：数值越小越趋向异常，数值越大越趋向正常，其中小于0.05表现为重度异常，小于0.1表现为中度异常，小于0.2表现为轻度异常，大于0.2表现为正常；\n右眼泪液分泌实验：数值越小越趋向异常，数值越大越趋向正常，其中小于10表现为异常，大于10表现为正常。'
#
#         if data[13] == '未发生排异':
#             r = '未发生排异'
#         else:
#             r = data[13]
#
#         cur_data['answer'] = f'{r}'
#
#         all_data.append(cur_data)
#
# # import codecs
# # f = codecs.open(save_f, "w", "utf-8-sig")
# # for i in all_data:
# #     i = json.dumps(i)
# #     f.write(i + '\n')
#
#
# with open(save_f, 'w', encoding='utf-8') as f:
#     for i in all_data:
#         i = json.dumps(i, ensure_ascii=False)
#         f.write(i + '\n')
