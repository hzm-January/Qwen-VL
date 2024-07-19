import xlrd, json

wb = xlrd.open_workbook('2024-6-6 最新扩增后AI图片诊断数据(1).xlsx') 
#按工作簿定位工作表
sh = wb.sheet_by_name('Sheet1')
print(11111111111, sh.nrows)#有效数据行数
print(222, sh.ncols)#有效数据列数
# print(sh.cell(0,0).value)#输出第一行第一列的值
# print(sh.row_values(0))#输出第一行的所有值
#将数据和标题组合成字典
print(3333, dict(zip(sh.row_values(0),sh.row_values(1))))
#遍历excel，打印所有数据

save_f = 'text_info_json.jsonl'

all_data=[]
for i in range(sh.nrows):
    print(55555, sh.row_values(i))
    
    # 组合为一段话
    cur_data = {}
    data = sh.row_values(i)
    
    
    if i>1:
        cur_data['姓名'] = data[0]
        cur_data['门诊时间'] = data[4] # 后面匹配图像后删除
        cur_data['性别'] = data[1]
        cur_data['年龄'] = data[2]        
        cur_data['眼别'] = data[5]
        print(type(data[3]))

        cur_data['眼表疾病指数量表'] = str(round(data[6], 2)) if isinstance(data[6], float) else ''  
        # if cur_data['眼表疾病指数量表']:
            # cur_data['眼表疾病指数量表']+='（眼表疾病指数量表：数值越大越趋向异常，数值越小越趋向正常，其中大于33表现为重度异常，大于23表现为中度异常，大于13表现为轻度异常，小于13表现为正常；）' 
        cur_data['角膜荧光染色评分'] = data[7] if isinstance(data[7], str) else str(data[7])
        # if cur_data['角膜荧光染色评分'] :
        #     cur_data['角膜荧光染色评分'] += '（角膜荧光染色评分：数值越大越趋向异常，数值越小越趋向正常，其中大于5表现为重度异常，大于3表现为中度异常，大于1表现为轻度异常，等于0表现为正常；）'
        cur_data['泪膜破裂时间'] = data[8] if isinstance(data[8], str) else str(data[8])
        # if cur_data['泪膜破裂时间']:
        #     cur_data['泪膜破裂时间'] += '（泪膜破裂时间：数值越小越趋向异常，数值越大越趋向正常，其中小于2表现为重度异常，小于5表现为中度异常，小于10表现为轻度异常，大于10表现为正常；）'
        cur_data['泪河高度'] = data[9] if isinstance(data[9], str) else str(data[9])
        # if cur_data['泪河高度'] :
        #     cur_data['泪河高度'] += '（泪河高度：数值越小越趋向异常，数值越大越趋向正常，其中小于0.05表现为重度异常，小于0.1表现为中度异常，小于0.2表现为轻度异常，大于0.2表现为正常；）'
        # cur_data['视力'] = data[10] if isinstance(data[10], str) else str(data[10])
        # cur_data['眼压'] = data[11] if isinstance(data[11], str) else str(data[11])
        cur_data['泪液分泌实验'] = data[12] if isinstance(data[12], str) else str(data[12])
        # if cur_data['泪液分泌实验']:
        #     cur_data['泪液分泌实验'] += '（泪液分泌实验：数值越小越趋向异常，数值越大越趋向正常，其中小于10表现为异常，大于10表现为正常。）'
        
        cur_data['您是否发生过皮肤排异'] = data[14]
        cur_data['您是否发生过口腔排异'] = data[15]
        cur_data['您是否发生过肠道排异'] = data[16]
        cur_data['您是否发生过肺排异'] = data[17]
        cur_data['您是否发生过肝排异'] = data[18]
        cur_data['哭时，是否有眼泪'] = data[19]
        cur_data['哭时有眼泪-流泪时感觉'] = data[20]
        cur_data['哭时无眼泪-无泪时感觉'] = data[21]
        cur_data['使用电子产品类型'] = data[22]
        cur_data['每天平均电子产品使用时间'] = data[23] if isinstance(data[23], str) else str(data[23])
        
        cnt = '在眼科诊断中，各个指标的严重程度分级为：\n眼表疾病指数量表：小于13正常，大于等于13且小于23轻度异常，大于等于23且小于33中度异常，大于等于33重度异常；\n右眼角膜荧光染色评分：0正常，1-2分轻度异常，3-4分中度异常，大于等于5分重度异常；\n右眼泪膜破裂时间：大于等于10s正常，6-10s轻度异常，2-5s中度异常，小于2s重度异常；\n右眼泪河高度：大于0.2正常，大于0.1且小于0.2轻度异常，大于0.05且小于等于0.1中度异常，小于等于0.05重度异常；\n右眼泪液分泌实验：大于10正常，小于等于10异常。'
        
        cnt = '在眼科诊断中，各个指标的严重程度分级为：\n眼表疾病指数量表：数值越大越趋向异常，数值越小越趋向正常，其中大于33表现为重度异常，大于23表现为中度异常，大于13表现为轻度异常，小于13表现为正常；\n右眼角膜荧光染色评分：数值越大越趋向异常，数值越小越趋向正常，其中大于5表现为重度异常，大于3表现为中度异常，大于1表现为轻度异常，等于0表现为正常；\n右眼泪膜破裂时间：数值越小越趋向异常，数值越大越趋向正常，其中小于2表现为重度异常，小于5表现为中度异常，小于10表现为轻度异常，大于10表现为正常；\n右眼泪河高度：数值越小越趋向异常，数值越大越趋向正常，其中小于0.05表现为重度异常，小于0.1表现为中度异常，小于0.2表现为轻度异常，大于0.2表现为正常；\n右眼泪液分泌实验：数值越小越趋向异常，数值越大越趋向正常，其中小于10表现为异常，大于10表现为正常。'
        
            
        
        if data[13]=='未发生排异':
            r = '未发生排异'
        else:
            r = data[13]
        
        cur_data['answer'] = f'{r}'
        
        all_data.append(cur_data)

# import codecs
# f = codecs.open(save_f, "w", "utf-8-sig")
# for i in all_data:
#     i = json.dumps(i)
#     f.write(i + '\n')


with open(save_f, 'w', encoding='utf-8') as f:
    for i in all_data:
        i = json.dumps(i,ensure_ascii=False)
        f.write(i + '\n')
