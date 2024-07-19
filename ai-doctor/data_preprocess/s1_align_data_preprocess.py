import json, sys, os, re
file = '/public/mmllm/caolili/bysy_all/data_bysy_latest2/text_info_json.jsonl'
all_data = {}
with open(file, 'r') as f:
    data = f.readlines()
    # print(data[:10])
    print(len(data))
    for data_i in data:
        d = json.loads(data_i)
        name = d['姓名']
        # time = d['门诊时间']
        # t_int_list = [str(int(i)) for i in d['门诊时间'].split(' ')[0].split('-')]
        # t = '-'.join(t_int_list)[2:]
        whichone = d['眼别']
        t_int_list = [str(int(i)) for i in d['门诊时间'].split(' ')[0].split('-')]
        t = '-'.join(t_int_list)[2:]
        # whichone = '右眼'
        
        
        # all_data[name] = d
        all_data[name + t + whichone] = d
print(all_data.keys())
# sys.exit(0)
save_f = 'bysy1_qwen_train_json_alignment.json'
save_f_test = 'bysy1_qwen_test_json.json'

new_save_data = []
test_all_data = []
num = 0
unvalid_num = 0
for path in ['/public/mmllm/caolili/bysy_all/bysy/data_v2']:
    for root, dirs, files in os.walk(path):
        # print(111111, root, dirs, files)
        if 'aug_data'  in root:
            continue
        if len(dirs)==0 and ('1.jpg' in files and '2.jpg' in files):
            # print(root, files)
                    
            assert os.path.exists(os.path.join(root, '1.jpg'))
            assert os.path.exists(os.path.join(root, '2.jpg'))
            
            name = os.path.basename(root)
            
            pattern_chinese = re.compile("[\u4e00-\u9fa5]+")
            pattern_time = re.compile("[\u4e00-\u9fa5]+(.*?)$")
            # print(11111111111, root,os.path.basename(root))
            
            patient_name = pattern_chinese.findall(os.path.basename(root))[0]
            patient_time = pattern_time.findall(os.path.basename(root))[0]
            print( patient_name, patient_time)
            
            # sys.exit(0)
            if patient_name+patient_time+'右眼' in all_data:
                
                data_json = all_data[patient_name+patient_time+'右眼']
                if '急性' in data_json['answer']:
                    continue
                    print(7777777777777777777777777777)
                
                # print(data_json)
                answer = data_json['answer']
                data_json.pop('姓名')
                # data_json.pop('门诊时间')
                data_json.pop('answer')
                # pritn(cnt)
                
                new_data = {}
                new_data['id'] = str(num)+'_'+patient_name+patient_time+'右眼'
                new_data['conversations'] = []
                new_data['type'] = 'stage2' # pt or sft 
                num += 1
                img1 = os.path.join(root, '1.jpg')
                img2 = os.path.join(root, '2.jpg')
                images = f'Picture 1: <img>{img1}</img>\n' + f'Picture 2: <img>{img2}</img>\n'
                value = '假设你是一个眼科专家，已知当前患者的检查结果与病史情况为：\n'+ str(data_json)
                
                
                cnt_alignment = f'本患者性别{data_json["性别"]}, 年龄{data_json["年龄"]}岁。经检测其{data_json["眼别"]}'
                x1 = data_json['眼表疾病指数量表']
                if x1 != '':
                    x1 = float(x1)
                    if x1<13:
                        r1='正常'
                    elif x1>=13 and x1<23:
                        r1='轻度异常'
                    elif x1>=23 and x1<33:
                        r1='中度异常'
                    elif x1>=33:
                        r1='重度异常'
                    else:
                        raise Error
                    cnt_alignment += f'眼表疾病指数量表{r1}, '
                        
                x2 = float(data_json['角膜荧光染色评分'])
                if x2==0:
                    r2='正常'
                elif x2<=2:
                    r2='轻度异常'
                elif x2<=4:
                    r2='中度异常'
                elif x2>=5:
                    r2='重度异常'
                else:
                    raise Error
                cnt_alignment+=f'角膜荧光染色评分{r2}, '

                x3 = data_json['泪膜破裂时间']
                if x3 !='':
                    x3 = float(x3)
                    if x3>=10:
                        r3='正常'
                    elif x3>=6 and x3<10:
                        r3='轻度异常'
                    elif x3>=2 and x3<6:
                        r3='中度异常'
                    elif x3<2:
                        r3='重度异常'
                    else:
                        raise Error
                    cnt_alignment+=f'泪膜破裂时间{r3}, '
                            
                x4 = float('0.3') if data_json['泪河高度']== '0. 3' else float(data_json['泪河高度']) 
                
                if x4>0.2:
                    r4='正常'
                elif x4>0.1 and x4<=0.2:
                    r4='轻度异常'
                elif x4>0.05 and x4<=0.1:
                    r4='中度异常'
                elif x4<=0.05:
                    r4='重度异常'
                else:
                    raise Error
                cnt_alignment+=f'泪河高度{r4}, '
                        
                
                x5 = data_json['泪液分泌实验']
                if x5 != '':
                    x5 = '35' if x5=='>35' else float(x5)
                    if isinstance(x5, str): x5=float(x5)
                    if x5>10:
                        r5='正常'
                    elif x5<=10:
                        r5='异常'
                    else:
                        raise Error
                    cnt_alignment+=f'泪液分泌实验{r5}, '
                
                
                if data_json['您是否发生过皮肤排异']== '是':
                    cnt_alignment += '发生过皮肤排异, '
                elif data_json['您是否发生过皮肤排异']=='否':
                    cnt_alignment += '未发生过皮肤排异, '
                if data_json['您是否发生过口腔排异']== '是':
                    cnt_alignment += '发生过口腔排异, '
                elif data_json['您是否发生过口腔排异']=='否':
                    cnt_alignment += '未发生过口腔排异, '
                if data_json['您是否发生过肠道排异']== '是':
                    cnt_alignment += '发生过肠道排异, '
                elif data_json['您是否发生过肠道排异']=='否':
                    cnt_alignment += '未发生过肠道排异, '
                if data_json['您是否发生过肺排异']== '是':
                    cnt_alignment += '发生过肺排异, '
                elif data_json['您是否发生过肺排异']=='否':
                    cnt_alignment += '未发生过肺排异, '
                if data_json['您是否发生过肝排异']== '是':
                    cnt_alignment += '发生过肝排异, '
                elif data_json['您是否发生过肝排异']=='否':
                    cnt_alignment += '未发生过肝排异, '
                if data_json['哭时，是否有眼泪']== '是':
                    cnt_alignment += '哭时有眼泪, '
                elif data_json['哭时，是否有眼泪']=='否':
                    cnt_alignment += '哭时无眼泪, '
                
                if data_json['哭时有眼泪-流泪时感觉']!='':
                    cnt_alignment += f'哭时{data_json["哭时有眼泪-流泪时感觉"]}, '
                if data_json['哭时无眼泪-无泪时感觉']!='':
                    cnt_alignment += f'哭时{data_json["哭时无眼泪-无泪时感觉"]}, '
                    
                if data_json['使用电子产品类型']!='':
                    cnt_alignment += f'使用电子产品类型为：{data_json["使用电子产品类型"]}，'
                
                if data_json['每天平均电子产品使用时间'] !='':
                    cnt_alignment += f'每天平均电子产品使用时间{data_json["每天平均电子产品使用时间"]}。'
                    
                if cnt_alignment[-2:]==', ':
                    # print(9999999999999999)
                    cnt_alignment = cnt_alignment[:-2]+'。'
                
                # value += '\n请根据图片与诊断信息，判断患者是否患有慢性或急性移植物抗宿主病。'
                
                # cnt = '\n在眼科诊断中，各个指标的严重程度分级为：\n眼表疾病指数量表：小于13正常，大于等于13且小于23轻度异常，大于等于23且小于33中度异常，大于等于33重度异常；角膜荧光染色评分：0正常，1-2分轻度异常，3-4分中度异常，大于等于5分重度异常；泪膜破裂时间：大于等于10s正常，6-10s轻度异常，2-5s中度异常，小于2s重度异常；泪河高度：大于0.2正常，大于0.1且小于0.2轻度异常，大于0.05且小于等于0.1中度异常，小于等于0.05重度异常；泪液分泌实验：大于10正常，小于等于10异常。'
                
                cnt = '已知在眼科诊断中，各个指标的严重程度分级为：\n眼表疾病指数量表：数值越大越趋向异常，数值越小越趋向正常，其中大于33表现为重度异常，大于23表现为中度异常，大于13表现为轻度异常，小于13表现为正常；\n角膜荧光染色评分：数值越大越趋向异常，数值越小越趋向正常，其中大于5表现为重度异常，大于3表现为中度异常，大于1表现为轻度异常，等于0表现为正常；\n泪膜破裂时间：数值越小越趋向异常，数值越大越趋向正常，其中小于2表现为重度异常，小于5表现为中度异常，小于10表现为轻度异常，大于10表现为正常；\n泪河高度：数值越小越趋向异常，数值越大越趋向正常，其中小于0.05表现为重度异常，小于0.1表现为中度异常，小于0.2表现为轻度异常，大于0.2表现为正常；\n泪液分泌实验：数值越小越趋向异常，数值越大越趋向正常，其中小于10表现为异常，大于10表现为正常。'
                

                cot = "\n已知鉴别是否患有慢性或急性移植物抗宿主病的依据为：眼表疾病指数量表、泪液分泌实验指标是否异常，结膜充血并伴裂隙灯显微镜下发现干燥性角膜结膜炎表现，并考虑本疾病系统性的存在与否。"
                # value += cot
                diff = '急性与慢性的发病机制不同：急性主要体现在结膜受累，很少出现角膜病变，发病早于慢性，持续一两周或一个月；慢性主要体现在结膜、泪腺、睑板腺的炎症和纤维化，以及角膜病变，发病晚但持续时间更长。'
                    
                diff2 = '急性与慢性的发病机制不同：急性主要是全身的细胞因子风暴，体现在结膜受累，很少出现角膜病变；慢性患者由于泪腺纤维化、睑板腺功能障碍和结膜杯状细胞缺失，导致泪液缺失及泪膜不稳定，从而导致严重的眼表干燥和进一步的角膜损伤，包括点状角膜病变、丝状角膜炎、角膜溃疡和穿孔、角膜血管化等，患者常有严重眼干、眼烧灼感、异物感、 畏光、视力下降等症状。'
                 
                            
                new_data['conversations'].append({
                                "from": 'user', 
                                "value": '<|extra_0|>'+value+'<|extra_1|>'  
                            })
                result = '检查结果为：'+ answer
                new_data['conversations'].append({
                                "from": 'assistant', 
                                "value": cnt_alignment
                            })
                # if '急性' in answer:                    
                #     new_save_data.append(new_data)  
                #     new_save_data.append(new_data)  
                #     new_save_data.append(new_data)   # 20条样本增加到80条
                
                
                
                new_save_data.append(new_data)  
                test_all_data.append({'image':[img1,img2],
                                'question':value +cnt+ cot , # + cnt + cot
                                'answer':result,
                                'question_id': new_data['id']
    })
                

            else:
                unvalid_num += 1
                print(8888888888888,patient_name+patient_time  )
        
f.close()
# 所有样本都用来做训练
with open(save_f, 'w', encoding='utf8') as f:
    json.dump(new_save_data, f, ensure_ascii=False, indent=4)
print(111111111111111, len(new_save_data))

# with open(save_f_test, 'w', encoding='utf8') as f:
#     for i in test_all_data[-52:]:
#         i = json.dumps(i, ensure_ascii=False)
#         f.write(i)
#         f.write('\n')

# with open(save_f_test, 'w', encoding='utf8') as f:  
#     json.dump(test_all_data[-52:], f, ensure_ascii=False, indent=4)
# print(111111111111111, len(test_all_data[-52:]))
        