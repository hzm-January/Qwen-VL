INT_MAX = 9999999999
INT_MIN = -9999999999
TYPE_INFO = 1
TYPE_INDC = 2
FLAG_TRUE = "是"
FLAG_FALSE = "否"
FLAG_NORM = "正常"
FLAG_ABNORM = "异常"

D_Mapping = {
    "DangerousFactor_Mapping": {
        "type": TYPE_INFO,  # TYPE_INFO,
        "rule": {
            "性别": {1: "性别男", 2: "性别女"},
            "睡觉时是否打鼾或患有睡眠呼吸暂停综合征？": {1: "睡觉时打鼾或患有睡眠呼吸暂停综合症", 2: "睡觉时不打鼾且没有睡眠呼吸暂停综合症"},
            "春季角结膜炎": {1: "有春季结膜炎患病史", 2: "没有春季结膜炎患病史"},
            "过敏性结膜炎": {1: "有过敏性结膜炎患病史", 2: "没有过敏性结膜炎患病史"},
            "倒睫": {1: "有倒睫症状", 2: "没有倒睫症状"},
            "干眼症": {1: "患有干眼症", 2: "没有干眼症"},
            "角膜炎": {1: "患有角膜炎", 2: "没有角膜炎"},
            "睑缘炎": {1: "患有睑缘炎", 2: "没有睑缘炎"},
            "眼睑松弛综合征": {1: "患有眼睑松弛综合征", 2: "没有眼睑松弛综合征"},
            "眼外伤": {1: "有眼外伤", 2: "没有眼外伤"},
            "是否患有过敏性疾病？": {1: "有过敏性疾病史", 2: "没有过敏性疾病史"},
            "是否对某些物质过敏？": {1: "有过敏史", 2: "没有过敏史"},
            "甲状腺疾病": {1: "患有甲状腺疾病", 2: "没有甲状腺疾病"},
            "是否患有其他疾病？": {1: "患有其他疾病", 2: "没有其他疾病"},
            "是否用过外源性性激素药物？": {1: "用过外源性激素药物", 2: "没有用过外源性激素药物"},
            "睡觉时是否偏好把手或手臂垫放在眼睛上？": {1: "睡觉时经常把手或手臂垫放在眼睛上", 2: "睡觉时不常把手或手臂垫放在眼睛上"},
            "常在大量灰尘环境中工作或生活？": {1: "经常在大量灰尘环境中工作或生活", 2: "不常在大量灰尘环境中工作或生活"},
            "常于夜间工作/学习？": {1: "经常在夜间工作或学习", 2: "不常在夜间工作或学习"},
            "感到工作/学习压力很大？": {1: "感到工作或学习压力很大", 2: "没有感到工作或学习压力很大"},
            "是否吸烟？": {1: "吸烟", 2: "不吸烟"},
            "是否饮酒？": {1: "饮酒", 2: "不饮酒"},
            "是否怀过孕？": {1: "怀过孕", 2: "没有怀过孕"},
            "惯用手": {1: "惯用手是左手", 2: "惯用手是右手"},
            "圆锥角膜家族史": {1: "有圆锥角膜家族患病史", 2: "没有圆锥角膜家族患病史"}
        }
    },
    "Pentacam_Mapping": {
        "type": TYPE_INDC,
        "rule": {
            "K1 F (D):": {"bins": [INT_MIN, 45, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "K2 F (D):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Km F (D):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "K Max (Front):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "K Max X (Front):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "K Max Y (Front):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "RMS (CF):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS HOA (CF):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS LOA (CF):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},

            "K1 B (D):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "K2 B (D):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "Km B (mm):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},

            "RMS (CB):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS HOA (CB):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS LOA (CB):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},

            "Pachy Apex:(CCT)": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "Pachy Min: (TCT)": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},

            "Dist. Apex-Thin.Loc. [mm]:(Dist. C-T)": {"bins": [INT_MIN, 8, INT_MAX],
                                                      "labels": [FLAG_NORM, FLAG_ABNORM]},
            "PachyMinX:": {},
            "PachyMinY:": {},
            "EccSph:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS (Cornea):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS HOA (Cornea):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS LOA (Cornea):": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},

            "C.Vol D 3mm:": {},
            "C.Vol D 5mm:": {},
            "C.Vol D 7mm:": {},
            "C.Vol D 10mm:": {},
            "BAD Df:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "BAD Db:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "BAD Dp:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "BAD Dt:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "BAD Da:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "BAD Dy:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "ISV:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "IVA:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "KI:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "CKI:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "IHA:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "IHD:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "R Min (mm)": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "Pachy Prog Index Min.:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Pachy Prog Index Max.:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Pachy Prog Index Avg.:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "ART Min.:": {},
            "ART Max.:": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "ART Avg.:": {},
            "C.Volume:(chamber volume)": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Chamber angle": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "A.C.Depth Int": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Elevation front": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Elevation back": {"bins": [INT_MIN, 8, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
        }
    },

    "Corvis_Mapping": {
        "type": TYPE_INDC,
        "rule": {
            "IOP [mmHg]": {},
            "DA [mm]": {},
            "A1T [ms]": {"bins": [INT_MIN, 7.15, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "A1V [m/s]": {"bins": [INT_MIN, 0.16, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "A2T [ms]": {"bins": [INT_MIN, 22.1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "A2V [m/s]": {"bins": [INT_MIN, -0.3, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "HCT [ms]": {"bins": [INT_MIN, 17.25, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "PD [mm]": {"bins": [INT_MIN, 5.3, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "A1L [mm]": {},
            "HCDL [mm]": {},
            "A2L [mm]": {},
            "A1DeflAmp. [mm](A1DA)": {"bins": [INT_MIN, 0.095, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "HCDeflAmp. [mm](HCDA)": {"bins": [INT_MIN, 1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "A2DeflAmp. [mm](A2DA)": {"bins": [INT_MIN, 0.11, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "A1DeflArea [mm^2]": {"bins": [INT_MIN, 0.19, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "HCDeflArea [mm^2]": {"bins": [INT_MIN, 4, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "A2DeflArea [mm^2]": {"bins": [INT_MIN, 0.25, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "A1ΔArcL [mm]": {},
            "HCΔArcL [mm]": {},
            "A2ΔArcL [mm]": {},
            "MaxIR [mm^-1]": {"bins": [INT_MIN, 0.18, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "DAR2": {"bins": [INT_MIN, 5, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "DAR1": {"bins": [INT_MIN, 1.55, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "ARTh": {"bins": [INT_MIN, 300, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "bIOP": {"bins": [INT_MIN, 14, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "IR [mm^-1]": {"bins": [INT_MIN, 9, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "SP A1": {"bins": [INT_MIN, 80, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "SSI": {"bins": [INT_MIN, 0.75, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
        }
    }
}
