INT_MAX = 9999999999
INT_MIN = -9999999999
TYPE_INFO = 1
TYPE_INDC = 2
FLAG_TRUE = "是"
FLAG_FALSE = "否"
FLAG_NORM = "normal"  # "正常"
FLAG_ABNORM = "abnormal"  # "异常"

W_Mapping = {
    "DangerousFactor_Mapping": {
        "type": TYPE_INFO,
        "rule": {
            "性别": {1: "男", 2: "女"},
            "睡觉时是否打鼾或患有睡眠呼吸暂停综合征？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "春季角结膜炎": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "过敏性结膜炎": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "倒睫": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "干眼症": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "角膜炎": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "睑缘炎": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "眼睑松弛综合征": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "眼外伤": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "是否患有过敏性疾病？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "是否对某些物质过敏？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "甲状腺疾病": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "是否患有其他疾病？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "是否用过外源性性激素药物？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "睡觉时是否偏好把手或手臂垫放在眼睛上？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "常在大量灰尘环境中工作或生活？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "常于夜间工作/学习？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "感到工作/学习压力很大？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "是否吸烟？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "是否饮酒？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "是否怀过孕？": {1: FLAG_TRUE, 2: FLAG_FALSE},
            "惯用手": {1: "左", 2: "右"},
            "圆锥角膜家族史": {1: FLAG_TRUE, 2: FLAG_FALSE}
        }
    },
    "Pentacam_Mapping": {
        "type": TYPE_INDC,
        "rule": {
            "K1 F (D):": {"bins": [INT_MIN, 45, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "K2 F (D):": {"bins": [INT_MIN, 49, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Km F (D):": {"bins": [INT_MIN, 46, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "K Max (Front):": {"bins": [INT_MIN, 48.2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "K Max X (Front):": {"bins": [INT_MIN, 0.07, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "K Max Y (Front):": {"bins": [INT_MIN, -1, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "RMS (CF):": {"bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS HOA (CF):": {"bins": [INT_MIN, 0.4, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS LOA (CF):": {"bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},

            "K1 B (D):": {"bins": [INT_MIN, -6.4, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "K2 B (D):": {"bins": [INT_MIN, -6.6, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "Km B (mm):": {"bins": [INT_MIN, -6.4, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},

            "RMS (CB):": {"bins": [INT_MIN, 1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS HOA (CB):": {"bins": [INT_MIN, 0.25, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS LOA (CB):": {"bins": [INT_MIN, 1.1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},

            "Pachy Apex:(CCT)": {"bins": [INT_MIN, 516, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "Pachy Min: (TCT)": {"bins": [INT_MIN, 511, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},

            "Dist. Apex-Thin.Loc. [mm]:(Dist. C-T)": {"bins": [INT_MIN, 0.9, INT_MAX],
                                                      "labels": [FLAG_NORM, FLAG_ABNORM]},
            "PachyMinX:": {},
            "PachyMinY:": {},
            "EccSph:": {"bins": [INT_MIN, 0.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS (Cornea):": {"bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS HOA (Cornea):": {"bins": [INT_MIN, 0.5, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "RMS LOA (Cornea):": {"bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},

            "C.Vol D 3mm:": {},
            "C.Vol D 5mm:": {},
            "C.Vol D 7mm:": {},
            "C.Vol D 10mm:": {},
            "BAD Df:": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "BAD Db:": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "BAD Dp:": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "BAD Dt:": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "BAD Da:": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "BAD Dy:": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "ISV:": {"bins": [INT_MIN, 41, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "IVA:": {"bins": [INT_MIN, 0.32, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "KI:": {"bins": [INT_MIN, 1.07, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "CKI:": {"bins": [INT_MIN, 1.03, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "IHA:": {"bins": [INT_MIN, 21, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "IHD:": {"bins": [INT_MIN, 0.016, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "R Min (mm)": {"bins": [INT_MIN, 7, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "Pachy Prog Index Min.:": {"bins": [INT_MIN, 0.88, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Pachy Prog Index Max.:": {"bins": [INT_MIN, 1.58, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Pachy Prog Index Avg.:": {"bins": [INT_MIN, 1.08, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "ART Min.:": {},
            "ART Max.:": {"bins": [INT_MIN, 412, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
            "ART Avg.:": {},
            "C.Volume:(chamber volume)": {"bins": [INT_MIN, 210, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Chamber angle": {"bins": [INT_MIN, 44, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "A.C.Depth Int": {"bins": [INT_MIN, 3.4, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Elevation front": {"bins": [INT_MIN, 6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
            "Elevation back": {"bins": [INT_MIN, 12.3, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
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
