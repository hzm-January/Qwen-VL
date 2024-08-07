import re

str = ("Assuming you are an ophthalmology specialist, "
       "the patient's diagnosis details indicate the following: "
       "K1 F (D): is normal,K2 F (D): is normal,Km F (D): is normal,"
       "Maximum keratometry of the front surface is normal,"
       "Steepest point of the front surface keratometry displacement in the x-axis is abnormal,"
       "Steepest point of the front surface keratometry displacement in the y-axis is normal,"
       "RMS (CF): is abnormal,RMS HOA (CF): is abnormal,"
       "RMS LOA (CF): is abnormal,"
       "K1 B (D): is normal,"
       "K2 B (D): is normal,"
       "Km B (mm): is normal,"
       "RMS (CB): is normal,"
       "Stress-strain index is normal,"
       "RMS HOA (CB): is normal,RMS LOA (CB): is normal,Pachy Apex:(CCT) is normal,Pachy Min: (TCT) is normal,Dist.")

# index = 'K1 F (D):'
index = 'Stress-strain index'
# a = re.findall(fr'({re.escape(index)}.*?),', str) # 为特殊字符添加转义符 如 (, ), [, ], ., *, +, ?, ^, $, {, }, |, \）
a = re.findall(fr"(Stress-strain.*?),", str)
print(a)
print(a[0].split('is'))