import torch, os, json
import argparse
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score

logging.basicConfig(format="[%(asctime)s]: %(message)s", datefmt="%Y-%M-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger('-')


def main(args):
    # dir_id = '20240725-104805'
    dir_id = args.dir_id

    base_model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/'
    lora_model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/diagnose/' + dir_id
    diagnose_test_dataset_json = lora_model_dir + '/train_test_dataset/diagnose_test_dataset.json'
    diagnose_test_label_json = lora_model_dir + '/train_test_dataset/diagnose_test_label.json'

    diagnose_test_dataset_dir = lora_model_dir + '/predict_result/'
    diagnose_predict_result_json = diagnose_test_dataset_dir + 'diagnose_predict_result.json'

    F_T_0 = 0  #
    F_T_1 = 1  # positive flag
    F_T_2 = 1  # positive flag
    F_T_3 = 1  # positive flag

    class_num = 4

    from peft import AutoPeftModelForCausalLM, PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,  # path to the output directory
        device_map="cuda:0",
        trust_remote_code=True,
        bf16=True
    )

    model = PeftModel.from_pretrained(base_model, lora_model_dir, device_map="cuda:0", trust_remote_code=True,
                                      bf16=True)

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
    logger.info("---- data count ----: %d", patient_cnt)

    correct = [0] * 4
    TP = [0] * class_num
    FP = [0] * class_num
    TN = [0] * class_num
    FN = [0] * class_num

    predicts = []
    label_cnt = [0] * class_num
    for i in range(patient_cnt):
        # print(diagnose_test_dataset[i])
        response, history = model.chat(tokenizer,
                                       query=diagnose_test_dataset[i],
                                       history=None)
        # print(response)
        # new_query = diagnose_test_dataset[i]+"请根据检测结果诊断该人员是否患有圆锥角膜病。"
        # response, history = model.chat(tokenizer, query=new_query, history=history)
        # label = F_T if label_info[i] else F_F
        # print(new_query)

        label = label_info[i]
        predict = 0
        if response == "forme fruste keratoconus":
            predict = 1
        elif response == "subclinical keratoconus":
            predict = 2
        elif response == "clinical keratoconus":
            predict = 3
        elif response == "No":
            predict = 0
        else:
            logger.warning(f'Error response : {response}')

        logger.info("id: %d, predict: %d, label: %d", i, predict, label)

        predicts.append(predict)

        if label == predict: correct[label] += 1
        label_cnt[label] += 1

    path = Path(diagnose_test_dataset_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    with open(diagnose_predict_result_json, 'w') as f:
        json.dump(predicts, f, ensure_ascii=False)

    logger.info(f"----    correct   ---- {correct}")
    logger.info(f"----  label count ---- {label_cnt}")
    accs = [corr / label_cnt[i] for i, corr in enumerate(correct)]
    logger.info(f"---- accuracy ---- {accs}")

    # 准确率
    accuracy = accuracy_score(label_info, predicts)
    logger.info("Accuracy: %f", accuracy)
    # 混淆矩阵
    conf_matrix = confusion_matrix(label_info, predicts)
    print("Confusion Matrix:\n", conf_matrix)
    # 分类报告
    class_report = classification_report(label_info, predicts)
    print("Classification Report:\n", class_report)
    # 精确率
    precision = precision_score(label_info, predicts, average='macro')
    logger.info("Precision: %f", precision)
    # 召回率
    recall = recall_score(label_info, predicts, average='macro')
    logger.info("Recall: %f", recall)
    # F1 分数
    f1 = f1_score(label_info, predicts, average='macro')
    logger.info("F1 Score: %f", f1)
    # ROC AUC
    # 计算ROC需要将模型输出概率归一化
    # prob_new = torch.nn.functional.softmax(torch.tensor(predicts), dim=1)
    # print(prob_new)
    # roc_auc = roc_auc_score(label_info, prob_new, average='macro', multi_class='ovo')
    # print("ROC AUC Score:", roc_auc)

    # sensitivity = TP / (TP + FN)
    # specificity = TN / (TN + FP)
    # # precision = TP/(TP+FP)
    # # recall = TP/(TP+FN)
    # # f1_ = 2*precision*recall/(precision+recall)
    #
    # f1 = 2 * TP / (2 * TP + FP + FN)
    # logger.info('TP: %d, FP: %d, TN: %d, FN: %d', TP, FP, TN, FN)
    # logger.info('准确率：%f %f', correct / patient_cnt)
    # logger.info('灵敏度：%f', sensitivity)
    # logger.info('特异度：%f', specificity)
    # logger.info('F1-Score：%f', f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test  checkpoint.")

    parser.add_argument(
        "-o", "--dir-id", type=str, default="20240725-104805"
    )

    args = parser.parse_args()

    main(args)
