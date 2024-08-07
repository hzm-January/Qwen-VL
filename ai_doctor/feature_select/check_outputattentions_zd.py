import re, os, sys
import torch
import argparse
import jsonlines, json, copy
import numpy as np
# import datasets
# from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationConfig

sys.path.append('/public/mmllm/caolili/Qwen-VL-old-bysy2/Qwen_VL_new2')
from qwen_generation_utils import make_context, get_stop_words_ids, decode_tokens

import statistics
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm

from collections import defaultdict 


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
def visualize_attention(multihead_attention,output_path="atten_map_1.png",title="Layer 5"):
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()# Shape: (n_tokens, n_tokens)
    
    # pooling the attention scores  with stride 20
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0)
    
    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(5, 5),dpi=400)

    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    # set the x and y ticks to 20x of the original


    ax = sns.heatmap(averaged_attention,
                cmap=cmap,  # custom color map
                norm=log_norm,  # 
                # cbar_kws={'label': 'Attention score'},
                )
    
    # remove the x and y ticks
    
    # replace the x and y ticks with string

    x_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    y_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_yticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # change the x tinks font size
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    
    # make y label vertical
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)     
    
    plt.title(title)
    # tight layout
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()

    top_five_attentions = []
    for row in averaged_attention:
        # Use torch.topk to get the top 5 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)
        
    return top_five_attentions,averaged_attention



# def generate_sample_ori(model, tokenizer, value, additive_input=''):
#     '''直接构建generate函数'''
#     f = '/public/mmllm/caolili/Qwen-VL-old-bysy2/Qwen_VL_new2'
#     generation_config = GenerationConfig.from_pretrained(f)

#     raw_text, context_tokens = make_context(
#             tokenizer,
#             query=value+additive_input,
#             history=[],
#             system="You are a helpful medical assistant.",
#             max_window_size=generation_config.max_window_size,
#             chat_format=generation_config.chat_format,
#         )
#     stop_words_ids = []
#     stop_words_ids.extend(get_stop_words_ids(
#             generation_config.chat_format, tokenizer
#         ))
#     input_ids = torch.tensor([context_tokens]).cuda()
#     outputs = model.generate(
#                 input_ids,
#                 stop_words_ids=stop_words_ids,
#                 generation_config=generation_config,
#                 max_new_tokens=100,
#                 return_dict_in_generate=True,
#                 output_attentions=True,
#                 output_scores=True,
#             )


def generate_sample(model, tokenizer, value, additive_input=''):
    '''参考chat()构建generate函数'''
    f = '/public/mmllm/caolili/Qwen-VL-old-bysy2/Qwen_VL_new2'
    generation_config = GenerationConfig.from_pretrained(f)

    raw_text, context_tokens = make_context(
            tokenizer,
            query=value+additive_input,
            history=[],
            system="You are a helpful medical assistant.",
            max_window_size=generation_config.max_window_size,
            chat_format=generation_config.chat_format,
        )
    # print(9999999999999999999100000000000,raw_text)
    print(len(context_tokens)) # 输入是1180个token
    stop_words_ids = []
    stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, tokenizer
        ))
    input_ids = torch.tensor([context_tokens]).cuda()
    outputs = model.generate(
                input_ids,
                stop_words_ids=stop_words_ids,
                generation_config=generation_config,
                max_new_tokens=100,
                return_dict_in_generate=True,
                output_attentions=True,
                output_scores=True,
            )
    
    
    # print(outputs.keys())
    
    response = decode_tokens(
        outputs['sequences'][0], # 长度916
        tokenizer,
        raw_text_len=len(raw_text),
        context_length=len(context_tokens),
        chat_format=generation_config.chat_format,
        verbose=False,
        errors='replace')
    print(outputs['sequences'][0][-13:]) # 108044; 38342
    print(11111111111, response) # 这个回复是对的？
    return response, outputs['attentions'], input_ids


    


def plot_attn(outputs_attention, model):
    total_layers = model.config.num_hidden_layers
    output_path = './'
    if not os.path.exists(output_path+'attn_maps/'):
        os.mkdir(output_path+'attn_maps/')
    # draw attention maps
    for i in outputs_attention:
        print(7777777777777777, len(i)) # 每一个样本，输出长度为57/100
        print(7777777777777777, len(i[-1])) # 第一个样本的第一个字符,输出长度均为32;为什么是32个长度？应该和层次是相关的；选择的是第一层？
        for j in range(0,total_layers): # 选择第一个样本的每一层
            print(444444444444,j,  # It looks like the code snippet you provided is incomplete and
            # does not perform any specific action. The code only contains a
            # single character 'i' and some comment symbols '
            i[0][j].shape) # ([1, 32, 1160, 1160]) # 第一个样本的第一个字符的attention维度
            top5_attention,average_attentions = visualize_attention(i[0][j].cpu(),output_path=output_path+"attn_maps/atten_map_"+str(j)+".png",title="Layer "+str(j+1)) # 一直选择i[0]


def get_want_attn(info, input_tokens, outputs_attention, model, tokenizer):
    input_tokens = input_tokens[0].cpu().numpy().tolist() # .numpy()
    
    print(99999999977, len(outputs_attention), len(outputs_attention[0]), len(outputs_attention[0][0])) # 1, 13, 32# 1 13 32 1 32 903 903 #  (layers=32, batch_size, num_heads=32, sequence_length, sequence_length)
    
    # 慢性/未 从index==4开始
    # attns = outputs_attention[4][-1].cpu() # 第一个样本输出的第一个字符的最后一层
    # averaged_attention = torch.mean(attns, axis=1)[0].float()# Shape: (n_tokens, n_tokens)
    attns = outputs_attention[4][-1].cpu() # 第一个样本输出的第一个字符的最后一层
    print(attns.shape)
    averaged_attention = torch.mean(attns, axis=1)[0].float()# Shape: (n_tokens, n_tokens)
    print(averaged_attention.shape)
    
    cur_input_ids = tokenizer(info[1:-1], return_tensors='pt').input_ids[0].cpu().numpy().tolist()# .numpy()
    # cur_input_ids = tokenizer("'角膜荧光染色评分': '2.0'", return_tensors='pt').input_ids[0].cpu().numpy().tolist()# .numpy()
    print(666666666666666, info, cur_input_ids)
    
    final = None
    max_c = 0
    last_num = 0
    for n, i in enumerate(input_tokens):
        correct_num = 0
        correct_list = []
        if n< (len(input_tokens)-len(cur_input_ids)):
            for j in range(len(cur_input_ids)):
                if input_tokens[n+j]==cur_input_ids[0+j]:
                    correct_num += 1
                    correct_list.append(1)
                else:
                    correct_list.append(0)
                    
        if max_c< correct_num:
            max_c = correct_num
            print(n, input_tokens[n:n+15])
            print(correct_list)

        if last_num<n:
            if correct_num >= len(cur_input_ids)-1 and sum(correct_list[:len(cur_input_ids)-1])== len(cur_input_ids)-1: 
            # if correct_num >= len(cur_input_ids)-1 or sum(correct_list[:len(cur_input_ids)//2+1])== len(cur_input_ids)//2+1: # BUG 
                final = n
                last_num = final
                print(8888888888888,'只有一个才对', n)
                break
        
        
        
    # 583
    print(333333333333, max_c, len(cur_input_ids))
    print(final)
    
    cur_true_input_ids = tokenizer(info[1:-1].split(':')[-1], return_tensors='pt').input_ids[0].cpu().numpy().tolist()# .numpy()
    
    # cur_attn = averaged_attention[-1][final:final+len(cur_input_ids)]
    cur_attn = averaged_attention[-1][final+len(cur_input_ids)-len(cur_true_input_ids):final+len(cur_input_ids)]
    # print(cur_attn)
    r = torch.sum(cur_attn)
    print(r)

    return r.item(), (final, max_c, cur_input_ids)
    
    
    
            
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="/public/mmllm/caolili/Qwen-VL-old-bysy2/output_model_2task/finetune-full-base-20240722-031504-bysy-5211-r8",
    )
    
    # /public/mmllm/caolili/Qwen-VL-old-bysy2/output_model_2task/finetune-full-base-20240722-031504-bysy-5211-r8
    
    # 8 19 23 26 28 29 30 31 40 46
    parser.add_argument("-f", "--sample-input-file", type=str, default='/public/mmllm/caolili/Qwen-VL-old-bysy2/data_bysy_latest2/alignment_data7/bysy2_qwen_test_sft_r19.json')
    # parser.add_argument("-f", "--sample-input-file", type=str, default='/public/mmllm/caolili/bysy3_deljixing/bysy2_qwen_test2_json_deljixing.json')
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="bysy_test.jsonl"
    )
    args = parser.parse_args()
    args.checkpoint_path = '/public/mmllm/caolili/Qwen-VL-old-bysy2/output_model_2task/finetune-full-base-20240722-123801-bysy-5211-r26' # 多任务模型的效果
    # args.checkpoint_path = '/public/mmllm/caolili/Qwen-VL-old-bysy2/output_model/finetune-full-base-20240711-173453-bysy_lastest7-sft-2211-r8' # 还是用旧的模型更好


    test = []
    with open(args.sample_input_file, 'r') as f:
        datas = f.readlines()
        print(len(datas))
        for i in datas:
            test.append(json.loads(i))
    

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, device_map="cuda", trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    # model.generation_config.do_sample = False

    tot_length = len(test)
    acc_res = []
    
    all_data = []
    input_tokens = []
    
    print(len(test)) # 应该是1才对

    all_result = defaultdict(list) 


    for doc_ori in test[:200]: 
        doc = copy.deepcopy(doc_ori)
        value = ''
        for n, i in enumerate(doc['image']):
            value += f'Picture {n+1}: <img>{i}</img>\n'
        value += doc['question'] 
        print(2222222222, value)
        
        
        completion, attn, inputids = generate_sample(model, tokenizer, value)
        # _, attn = generate_sample(model, tokenizer, value, completion) # 不需要把输出也加上
        # input_tokens.append(inputids)
        
        answer = doc["answer"]
        # acc = is_correct(completion, answer)
        acc = 1
        doc["prediction"] = completion # 模型prediction
        doc["acc"] = acc
        doc.pop('image')
        # f_output.write(doc)
        acc_res.append(acc)
        all_data.append(doc)     

        test_lit = []
        # r = re.findall(r"('患者姓名'.*?), ",doc['question'])[0]; test_lit.append(r)
        # r = re.findall(r"('门诊时间'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('性别'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('年龄'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('眼别'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('眼表疾病指数量表'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('角膜荧光染色评分'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('泪膜破裂时间'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('泪河高度'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('泪液分泌实验'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('您是否发生过皮肤排异'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('您是否发生过口腔排异'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('您是否发生过肠道排异'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('您是否发生过肺排异'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('您是否发生过肝排异'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('哭时，是否有眼泪'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('哭时有眼泪-流泪时感觉'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('哭时无眼泪-无泪时感觉'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('使用电子产品类型'.*?), ",doc['question'])[0]; test_lit.append(r)
        r = re.findall(r"('每天平均电子产品使用时间'.*?)}",doc['question'])[0]; test_lit.append(r)
        # print(test_lit)
        

        all_infos = {}
        for i in test_lit:
            kk = i.split(':')[0]
            
            r, info = get_want_attn(i, inputids, attn, model, tokenizer)
            all_result[kk].append(r)
            all_infos[kk] = info
            print('----------------------------------------------------------------------------', kk, i, r)
        with open(args.sample_output_file, 'w', encoding='utf8') as f:  
            json.dump(all_data, f, ensure_ascii=False, indent=4)

        # f_output.close()
        print("Acc: ", np.mean(acc_res))
        gc.collect()

        
    print(111111111111, all_infos)
    # for i in all_result.keys():
    #     print(i)
    #     average = statistics.mean(all_result[i])*100
    #     print(average)

    print_data = {}
    for i in all_result.keys():
        # print(i)
        average = statistics.mean(all_result[i])
        # print(round(average*100, 2))
        print_data[i] = round(average*100, 2)
    print(print_data)
    
    print(list(print_data.keys()))
    print(list(print_data.values()))
