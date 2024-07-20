import json, yaml, os
import pandas as pd


def handle_dangerous_factors(conf):
    # 2 open and load xlsx (require xlrd==1.2.0)
    df = pd.read_excel(conf['datasource_xlsx_path'], sheet_name=0)
    print('sheet index: %d, row count: %s, column count: %s' % (0, df.shape[0], df.shape[1]))  # sheet info
    # print(df.values[0, :])
    col_0_label = df.iloc[:, 0]
    col_x_data = df.iloc[:, 1:]
    # col_x_data.to_json(conf['xlsx2json_output_save_path'], orient='records', force_ascii=False)


def handle_sheet1(wb, conf):
    print()


def handle_sheet2(wb, conf):
    print()


def main():
    conf_path = r'/config/s1_align_conf.yaml'
    # 1 load config
    with open(os.path.dirname(__file__) + conf_path, 'r') as s1_yaml:
        s1_conf = yaml.safe_load(s1_yaml)
    file_pathes = s1_conf['file_path']
    file_names = s1_conf['file_name']
    load_path = file_pathes['load']
    save_path = file_pathes['save']

    am_df = pd.read_excel(load_path + file_names['abbr_mapping'], sheet_name=0)
    abbr_map = dict(zip(am_df[am_df.columns[0]], am_df[am_df.columns[1]]))
    # print(abbr_map)
    df = pd.read_excel(load_path + file_names['data'], sheet_name=None)
    sheet_names = list(df.keys())
    for sheet_name in sheet_names:
        sh_df = df[sheet_name]
        print(sh_df.columns)
        sh_df.columns = [abbr_map.get(col, col) if not pd.isna(abbr_map.get(col)) else col for col in sh_df.columns]
        print(sh_df.columns)
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
