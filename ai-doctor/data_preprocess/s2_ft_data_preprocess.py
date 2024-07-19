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
save_f = 'bysy1_qwen_train_json_sft.json'
# save_f_test = 'bysy1_qwen_test_json.json'

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
                new_data['type'] = 'stage3' # pt or sft 
                num += 1
                img1 = os.path.join(root, '1.jpg')
                img2 = os.path.join(root, '2.jpg')
                images = f'Picture 1: <img>{img1}</img>\n' + f'Picture 2: <img>{img2}</img>\n'
                value = '假设你是一个眼科专家，已知当前患者的检查结果与病史情况为：\n'+ str(data_json)
                
                question = '\n请根据图片与诊断信息，判断患者是否患有慢性移植物抗宿主病。'

                
                # cot2 = '\n已知诊断是否患有慢性移植物抗宿主病的依据为：眼表疾病指数量表、角膜荧光染色评分、泪河高度、泪膜破裂时间、泪液分泌实验等指标是否异常，裂隙灯显微镜下发现干燥性角膜结膜炎表现（结膜充血、水肿、睑板腺腺口堵塞、荧光素钠染色后角膜上皮点染），同时考虑哭时是否泪液减少、全身排异的存在与否、电子产品的使用情况。'  
                
                cot2 = '\n已知诊断是否患有慢性移植物抗宿主病的依据为：眼表疾病指数量表、角膜荧光染色评分、泪河高度、泪膜破裂时间、泪液分泌实验等指标是否异常，裂隙灯显微镜下是否发现干燥性角膜结膜炎表现（结膜充血、水肿、荧光素钠染色后角膜上皮点染等），同时考虑哭时是否有眼泪等病史情况，以及本疾病系统性的存在与否。'  
                            
                new_data['conversations'].append({
                                "from": 'user', 
                                "value": images+'<|extra_0|>'+value+'<|extra_1|>'+ question+cot2
                            })
                result = '检查结果为：'+ answer
                new_data['conversations'].append({
                                "from": 'assistant', 
                                "value": result
                            })
                # if '急性' in answer:                    
                #     new_save_data.append(new_data)  
                #     new_save_data.append(new_data)  
                #     new_save_data.append(new_data)   # 20条样本增加到80条
                
                
                
                new_save_data.append(new_data)  
                # test_all_data.append({'image':[img1,img2],
                #                 'question':value +cnt+ cot , # + cnt + cot
                #                 'answer':result,
                #                 'question_id':'bysy-'+str(num)})
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
        