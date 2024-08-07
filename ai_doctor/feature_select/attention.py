import argparse
import json
import re
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, PeftModel
from loguru import logger
from transformers.generation import GenerationConfig

token_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/'
base_model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/qwen-dpo/input-model/v2'
dpo_model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/qwen-dpo/output-model/result_10'

diagnose_test_dataset_json = '/data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/data/data_finetune/dpo/dpo_test_dataset.json'
diagnose_test_label_json = '/data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/data/data_finetune/dpo/dpo_test_label.json'

column_name_json = '/data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/data/data_sources/org_data/patient_infos_column_name_1_2.json'

generation_config_dir = '/data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/feature_select/'

cuda = "cuda:1"


def main(args):
    # 1 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(token_dir, trust_remote_code=True)

    generation_config = GenerationConfig.from_pretrained(generation_config_dir)
    logger.info(f'generation config: {generation_config}')

    # 2 model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map=cuda,
        torch_dtype="auto",
        trust_remote_code=True,
        bf16=True
    )

    model = PeftModel.from_pretrained(base_model, dpo_model_dir)
    model.eval()

    # 3 test dataset
    with open(diagnose_test_dataset_json, 'r') as file:
        diagnose_test_dataset = json.load(file)
    with open(diagnose_test_label_json, 'r') as file:
        label_info = json.load(file)

    # 获取attention
    with open(column_name_json, 'r') as f:
        column_names = json.load(f)

    # 4
    total_len = len(label_info)
    for i in range(total_len):
        query_str = diagnose_test_dataset[i]
        logger.info(f'-------- query_str: {query_str}')
        logger.info(f'-------- query_str length: {len(query_str)}')
        query_token = tokenizer(query_str, return_tensors='pt').to(cuda)  # 转到GPU上
        logger.info(f'-------- query_token: {query_token}')
        logger.info(f'-------- input_ids: {query_token["input_ids"]}')
        logger.info(f'-------- input_ids shape: {query_token["input_ids"].shape}')
        logger.info(f'-------- token_type_ids: {query_token["token_type_ids"]}')
        logger.info(f'-------- token_type_ids shape: {query_token["token_type_ids"].shape}')
        logger.info(f'-------- attention_mask: {query_token["attention_mask"]}')
        logger.info(f'-------- attention_mask shape: {query_token["attention_mask"].shape}')
        # logger.info(f'-------- offset mapping: {query_token["offsets_mapping"]}')

        input_ids_tensor = query_token["input_ids"]
        input_ids_len = input_ids_tensor.shape[1]
        input_ids = input_ids_tensor[0].tolist()

        logger.info(f'input_ids: {input_ids}')

        # stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]

        output = model.generate(
            **query_token,
            # stop_words_ids=stop_words_ids,
            generation_config=generation_config,
            # max_new_tokens=100,
            return_dict_in_generate=True,
            output_attentions=True,
            output_scores=True,
        )
        logger.info(f'-------- output type: {type(output)}')
        logger.info(f'-------- output sequences type: {type(output["sequences"])}')
        logger.info(f'-------- output sequences shape: {output["sequences"].shape}')
        logger.info(f'-------- output sequences[0] shape: {output["sequences"][0].shape}')
        logger.info(f'-------- output sequences[0]: {output["sequences"][0]}')
        attentions = output["attentions"]
        # logger.info(f'-------- output attention: {attentions}')
        # logger.info(f'-------- output attention length: {len(attentions)}')
        # logger.info(f'-------- output attention [-1]: {attentions[-1]}')
        logger.info(f'-------- output attention [-1] length: {len(attentions[-1])}') # 取输出的最后一个token对应的attentions
        logger.info(f'-------- output attention [-1][-1]: {attentions[-1][-1]}') # 取输出的最后一个token对应的最后一层attention
        logger.info(f'-------- output attention [-1][-1] shape: {attentions[-1][-1].shape}') # 取输出的最后一个token对应的最后一层attention

        attention = torch.mean(attentions[-1][-1], axis=1)[0].float()
        logger.info(f'-------- attention: {attention}')
        logger.info(f'-------- attention shape: {attention.shape}')

        output_token_decode_str = tokenizer.decode(output["sequences"][0], skip_special_tokens=True)
        logger.info(f'-------- output_str: {output_token_decode_str}')
        logger.info(f'-------- output_str len: {len(output_token_decode_str)}')

        answer = output_token_decode_str[len(query_str) - 1:]
        logger.info(f'-------- answer: {answer}')

        attn_map = {}

        for i, column_name in enumerate(column_names):
            if 'Stress' in column_name: continue  # TODO: 待修复，Stress-strain 匹配不到
            cur_strs = re.findall(fr'({re.escape(column_name)}.*?),', query_str)
            logger.info(f'{i}----{column_name}-----{cur_strs[0]}')
            cur_str = cur_strs[0]
            cur_str_tokens = tokenizer(cur_str, return_tensors='pt').to(cuda)
            cur_str_ids = cur_str_tokens["input_ids"][0].tolist()
            # logger.info(f'----cur_str_ids: {cur_str_ids}')
            cur_str_ids_len = len(cur_str_ids)
            start_i = -1
            for j in range(input_ids_len - cur_str_ids_len + 1):
                if input_ids[j:j + cur_str_ids_len] == cur_str_ids:
                    start_i = j
                    break

            # TODO: 待解决，B,H等开头的英文字母会和前面的逗号一起作为token
            if start_i == -1:
                cur_str_tokens = tokenizer(','+cur_str, return_tensors='pt').to(cuda)
                cur_str_ids = cur_str_tokens["input_ids"][0].tolist()
                # logger.info(f'----cur_str_ids: {cur_str_ids}')
                cur_str_ids_len = len(cur_str_ids)
                for j in range(input_ids_len - cur_str_ids_len + 1):
                    if input_ids[j:j + cur_str_ids_len] == cur_str_ids:
                        start_i = j
                        break
            # TODO: 待解决，B,H等开头的英文字母会和前面的逗号一起作为token
            if start_i == -1:
                cur_str_tokens = tokenizer(' ' + cur_str, return_tensors='pt').to(cuda)
                cur_str_ids = cur_str_tokens["input_ids"][0].tolist()
                # logger.info(f'----cur_str_ids: {cur_str_ids}')
                cur_str_ids_len = len(cur_str_ids)
                for j in range(input_ids_len - cur_str_ids_len + 1):
                    if input_ids[j:j + cur_str_ids_len] == cur_str_ids:
                        start_i = j
                        break
            print(start_i)
            attn = attention[-1][start_i: start_i+cur_str_ids_len]
            logger.info(f'-------- attn: {attn}')
            attn_sum = torch.sum(attn).item()
            attn_map[column_name] = attn_sum

        sorted_items = sorted(attn_map.items(), key=lambda item: item[1], reverse=False)
        sorted_attn_map = {key: value for key, value in sorted_items}
        logger.info(f'-------- attention: {sorted_attn_map}')

        # keys, values = zip(*sorted_items[:30])
        keys, values = zip(*sorted_items)
        # 绘制条形图
        plt.figure(figsize=(20, 50))
        bars = plt.barh(keys, values, color='blue')

        # 在条形图上显示数值
        for bar in bars:
            plt.text(
                bar.get_width(),  # X 坐标（条形的宽度，即数值）
                bar.get_y() + bar.get_height() / 2,  # Y 坐标（条形的中点）
                f'{bar.get_width()}',  # 显示的文本（数值）
                va='center',  # 垂直对齐方式
                ha='left',  # 水平对齐方式
                color='black',  # 文本颜色
                fontsize=8  # 文本字体大小
            )
        plt.xlabel('Values', fontsize=8, fontweight='bold', color='black')
        plt.ylabel('Items', fontsize=8, fontweight='bold', color='black')
        plt.title('Bar Chart of Values Sorted by Rank')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        # plt.subplots_adjust(left=0.5, right=0.95, top=0.95, bottom=0.1)
        plt.tight_layout()
        # 显示条形图
        plt.show()
        break

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Attention.")

    parser.add_argument(
        "-o", "--dir-id", type=str, default="20240725-104805"
    )

    args = parser.parse_args()

    main(args)
