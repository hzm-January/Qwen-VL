import json, math
import logging

import numpy as np

DDOF = 0

dir_ids = ['20240725-104805', '20240725-143158']
base_model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/'
lora_model_dir_pre = f'/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/diagnose/'

diagnose_test_dataset_json = '/train_test_dataset/diagnose_test_dataset.json'
diagnose_test_label_json = '/train_test_dataset/diagnose_test_label.json'
diagnose_predict_result_json = '/predict_result/diagnose_predict_result.json'

F_T = 1  # positive flag
F_F = 0  # negtive flag

available_files = []

true_positive = []  # GT:正 P:正 TP
false_positive = []  # GT:负 P:正 FP
true_negative = []  # GT:负 P:负 TN
false_negative = []  # GT:正 P:负 FN

result_all = []  # 多次采样的准确率
sensitivity_all = []  # 多次采样的敏感度
specificity_all = []  # 多次采样的特异度
f1_score_all = []  # 多次采样的F1


logging.basicConfig(level=logging.INFO,format="[%(asctime)s]: %(message)s", datefmt="%Y-%M-%d %H:%M:%S")
logger = logging.getLogger('-')

for dir_id in dir_ids:
    logger.info(lora_model_dir_pre + dir_id + diagnose_predict_result_json)

    with open(lora_model_dir_pre + dir_id + diagnose_test_dataset_json, 'r') as file:
        diagnose_test_dataset = json.load(file)
    with open(lora_model_dir_pre + dir_id + diagnose_test_label_json, 'r') as file:
        label_info = json.load(file)
    with open(lora_model_dir_pre + dir_id + diagnose_predict_result_json, 'r') as file:
        pred_info = json.load(file)

    patient_cnt = len(label_info)

    logger.info("---- data count ----: %d", patient_cnt)

    correct = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(patient_cnt):

        label, predict = label_info[i], pred_info[i]
        label = 1 if label else 0

        # print("id:", i, "predict: ", predict, "label: ", label)

        if label == predict: correct += 1

        if label == F_T and predict == F_T:
            TP += 1
        elif label == F_F and predict == F_T:
            FP += 1
        elif label == F_F and predict == F_F:
            TN += 1
        elif label == F_T and predict == F_F:
            FN += 1
        else:
            logging.INFO('Prediction is not Yes and not No either, It is ', predict, ', GT is ', label)

    true_positive.append(TP)
    false_positive.append(FP)
    true_negative.append(TN)
    false_negative.append(FN)

    sensitivity = TP / (TP + FN)
    sensitivity_all.append(sensitivity)
    specificity = TN / (TN + FP)
    specificity_all.append(specificity)
    # precision = TP/(TP+FP)
    # recall = TP/(TP+FN)
    # f1_ = 2*precision*recall/(precision+recall)

    f1 = 2 * TP / (2 * TP + FP + FN)
    f1_score_all.append(f1)

    if (correct / patient_cnt) >= 0.95 or specificity > 0.8:
        available_files.append(file)

    result_all.append(correct / patient_cnt)
    print('当前准确率：', correct / patient_cnt, (TP + TN) / (TP + FP + TN + FN))
    print('当前灵敏度， 特异度, F1', sensitivity, specificity, f1)

print(result_all)

print('平均准确率：', np.mean(result_all))
print('方差：', np.var(result_all))
print('标准差：', np.std(result_all, ddof=0))

print('平均灵敏度', np.mean(sensitivity_all))
print('特异度', np.mean(specificity_all))
print('F1score', np.mean(f1_score_all))

sample_num = len(dir_ids)

# 计算95%置信区间
print('---------准确率')
bound1 = np.mean(result_all) + 1.96 * (np.std(result_all, ddof=DDOF) / math.sqrt(sample_num))
bound2 = np.mean(result_all) - 1.96 * (np.std(result_all, ddof=DDOF) / math.sqrt(sample_num))
print(bound1)
print(bound2)
print("{:.4f}".format(np.mean(result_all)), "{:.4f}".format(bound2), "{:.4f}".format(bound1),
      "{:.4f}".format(bound1 - bound2))

# 计算95%置信区间
print('---------灵敏度')
bound1 = np.mean(sensitivity_all) + 1.96 * (np.std(sensitivity_all, ddof=DDOF) / math.sqrt(sample_num))
bound2 = np.mean(sensitivity_all) - 1.96 * (np.std(sensitivity_all, ddof=DDOF) / math.sqrt(sample_num))
print("{:.4f}".format(np.mean(sensitivity_all)), "{:.4f}".format(bound2), "{:.4f}".format(bound1),
      "{:.4f}".format(bound1 - bound2))

# 计算95%置信区间
print('---------特异度')
bound1 = np.mean(specificity_all) + 1.96 * (np.std(specificity_all, ddof=DDOF) / math.sqrt(sample_num))
bound2 = np.mean(specificity_all) - 1.96 * (np.std(specificity_all, ddof=DDOF) / math.sqrt(sample_num))
print("{:.4f}".format(np.mean(specificity_all)), "{:.4f}".format(bound2), "{:.4f}".format(bound1),
      "{:.4f}".format(bound1 - bound2))

# 计算95%置信区间
print('---------F1 score')
bound1 = np.mean(f1_score_all) + 1.96 * (np.std(f1_score_all, ddof=DDOF) / math.sqrt(sample_num))
bound2 = np.mean(f1_score_all) - 1.96 * (np.std(f1_score_all, ddof=DDOF) / math.sqrt(sample_num))
print("{:.4f}".format(np.mean(f1_score_all)), "{:.4f}".format(bound2), "{:.4f}".format(bound1),
      "{:.4f}".format(bound1 - bound2))

print("available_files_count: ", len(available_files))

print("files_count: ", len(dir_ids))
