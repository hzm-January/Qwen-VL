import torch, os, json
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)

dir_id = '20240725-104805'

base_model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/'
lora_model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/diagnose/' + dir_id

diagnose_test_dataset_json = lora_model_dir + '/train_test_dataset/diagnose_test_dataset.json'
diagnose_test_label_json = lora_model_dir + '/train_test_dataset/diagnose_test_label.json'

diagnose_predict_result_json = lora_model_dir + '/predict_result/diagnose_predict_result.json'

F_T = "Yes"  # positive flag
F_F = "No"  #

from peft import AutoPeftModelForCausalLM, PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_dir,  # path to the output directory
    device_map="cuda:0",
    trust_remote_code=True,
    bf16=True
)

model = PeftModel.from_pretrained(base_model, lora_model_dir, device_map="cuda:0", trust_remote_code=True, bf16=True)

model.eval()
# merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary.
# They respectively work for sharding checkpoint and save the model to safetensors

tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
if not hasattr(tokenizer, 'model_dir'):
    tokenizer.model_dir = base_model_dir

with open(diagnose_test_dataset_json, 'r') as file:
    diagnose_test_dataset = json.load(file)
with open(diagnose_test_label_json, 'r') as file:
    label_info = json.load(file)

patient_cnt = len(label_info)
print("---- data count ----: ", patient_cnt)

pred = []
for i in range(patient_cnt):
    # print(diagnose_test_dataset[i])
    response, history = model.chat(tokenizer,
                                   query=diagnose_test_dataset[i],
                                   history=None)
    # print(response)
    # new_query = diagnose_test_dataset[i]+"请根据检测结果诊断该人员是否患有圆锥角膜病。"
    # response, history = model.chat(tokenizer, query=new_query, history=history)
    ass_value = F_T if label_info[i] else F_F
    # print(new_query)
    print("id:", i, "predict: ", response, "label: ", ass_value)
    pred.append(1 if response == F_T else 0)

with open(diagnose_predict_result_json, 'w') as f:
    json.dump(pred, f, ensure_ascii=False)
