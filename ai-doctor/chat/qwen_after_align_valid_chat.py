import torch, os
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)

model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/align/20240721-211309/'

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if not hasattr(tokenizer, 'model_dir'):
    tokenizer.model_dir = model_dir
# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
# 默认gpu进行推理，需要约24GB显存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()

# 指定生成超参数（transformers 4.32.0及以上无需执行此操作）
# model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

# 第一轮对话
# Either a local path or an url between <img></img> tags.
# image_path = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
# query = (
#     "假设你是一个眼科专家，已知当前患者的检查结果与病史情况为：\n{"
#     "'性别': '男', '年龄': '61', '眼别': '左眼', '眼表疾病指数量表': '43.18', '角膜荧光染色评分': '0', '泪膜破裂时间': '3', '泪河高度': '0.2', "
#     "'泪液分泌实验': '6', '您是否发生过皮肤排异': '是', '您是否发生过口腔排异': '是', '您是否发生过肠道排异': '否', '您是否发生过肺排异': '否', "
#     "'您是否发生过肝排异': '否', '哭时，是否有眼泪': '是', '哭时有眼泪-流泪时感觉': '1', '哭时无眼泪-无泪时感觉': '1', '使用电子产品类型': '手机', "
#     "'每天平均电子产品使用时间': ''}\n")

query = (
    "<|extra_0|>假设你是一个眼科专家，已知当前患者的检查结果与病史情况为：\n{"
    "'性别': '男', '年龄': '61', '眼别': '左眼', '眼表疾病指数量表': '43.18', '角膜荧光染色评分': '0', '泪膜破裂时间': '3', '泪河高度': '0.2', "
    "'泪液分泌实验': '6', '您是否发生过皮肤排异': '是', '您是否发生过口腔排异': '是', '您是否发生过肠道排异': '否', '您是否发生过肺排异': '否', "
    "'您是否发生过肝排异': '否', '哭时，是否有眼泪': '是', '哭时有眼泪-流泪时感觉': '1', '哭时无眼泪-无泪时感觉': '1', '使用电子产品类型': '手机', "
    "'每天平均电子产品使用时间': '10'}<|extra_1|>\n"
    "请根据诊断信息，判断患者是否患有慢性移植物抗宿主病。\n"
    "已知诊断是否患有慢性移植物抗宿主病的依据为：眼表疾病指数量表、角膜荧光染色评分、泪河高度、泪膜破裂时间、泪液分泌实验等指标是否异常，裂隙灯显微镜下是否发现干燥性角膜结膜炎表现"
    "（结膜充血、水肿、荧光素钠染色后角膜上皮点染等），同时考虑哭时是否有眼泪等病史情况，以及本疾病系统性的存在与否。")

response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，与人互动。

# # 第二轮对话
# response, history = model.chat(tokenizer, '输出击掌的检测框', history=history)
# print(response)
# # <ref>"击掌"</ref><box>(211,412),(577,891)</box>
# image = tokenizer.draw_bbox_on_latest_picture(response, history)
# if image:
#     image.save('output_chat.jpg')
# else:
#     print("no box")
