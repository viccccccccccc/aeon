import numpy as np
import scipy.io
import pdb
import pickle

from aeon.transformations.spikelet.spikelet_approx import Spikelet_aproximation_ver_03
from aeon.transformations.spikelet.spikelet_char_query import Spikelet_Char_query
from aeon.transformations.spikelet.spikelet_matprof import Spikelet_MP_new_ver_02
from aeon.transformations.spikelet.spikelet_params import (
    Spikelet_MpParam_generate_ver_02,
)
from aeon.transformations.spikelet.spikelet_word_query import Spikelet_Word_query


# Define a class for Spikelet algorithm parameters
class AlgParam:
    def __init__(self):
        self.DataName = "psyllid"
        self.DataColumn = 1
        self.query = ["B"]
        self.supp_max = 200
        self.supp_min = 50
        self.operation = {
            "operation_sequence": [
                "restrictSupportByWindowLength",
                "reduceSpikeByMagnitude",
                "restrictSupportByMagnitudeRatioInitial",
                "extractConstantSegment"
            ],
            "restrictSupportByMagnitudeRatio": {"magnitude_ratio": 0}
        }
        self.symbol_mapping = {
            "rule": [
                {"condition": "Type == 2 and SuppCBM_100 >= 50", "symbol": "B"},
                {"condition": "Type == 2 and SuppCBM_100 < 50", "symbol": "A"}
            ],
            "argument": [2, "SuppCBM_100", 50, ["B", "A"]]
        }

def Spikelet_CallMpExec_getParam(AlgParam, EnvParam, ThrParam=None):
    # Set DataFileName if specified in ThrParam
    if ThrParam and 'DataFileName' in ThrParam:
        AlgParam.DataFileName = ThrParam['DataFileName']
    else:
        AlgParam.DataFileName = AlgParam.DataName

    # Set magnitude and constant length thresholds if available
    if ThrParam and 'MaT' in ThrParam:
        AlgParam.magnitude_threshold = ThrParam['MaT']
    if ThrParam and 'CoT' in ThrParam:
        AlgParam.constant_length_threshold = ThrParam['CoT']

    # Set environment parameters based on the option in ThrParam
    Option = ThrParam.get('exec', 'execonly') if ThrParam else 'execonly'
    EnvParam = CallMpExec_getEnvParam(AlgParam, EnvParam, Option)

    # Set MagInfo_File if specified
    if ThrParam and 'MagInfo_File' in ThrParam:
        EnvParam.MagInfo_File = ThrParam['MagInfo_File']

    # Handle MpPlot fields if specified
    if ThrParam and 'MpPlot' in ThrParam:
        for fn_i in ThrParam['MpPlot']:
            EnvParam.MpPlot[fn_i] = ThrParam['MpPlot'][fn_i]

    return AlgParam, EnvParam

def CallMpExec_getEnvParam(AlgParam, EnvParam, Option=None):
    if Option == 'all':
        EnvParam = get_EnvParam_all(AlgParam, EnvParam)
    elif Option == 'plotonly':
        EnvParam = get_EnvParam_plotonly(AlgParam, EnvParam)
    elif Option == 'test':
        EnvParam = get_EnvParam_test(AlgParam, EnvParam)
    elif Option == 'execonly':
        EnvParam = get_EnvParam_execonly(AlgParam, EnvParam)
    elif Option == 'testonly':
        EnvParam = get_EnvParam_testonly(AlgParam, EnvParam)
    else:
        EnvParam = get_EnvParam_plotonly(AlgParam, EnvParam)
    return EnvParam

def get_EnvParam_all(AlgParam, EnvParam):
    # Setting full environment parameters
    EnvParam.Data_Dir = f"{EnvParam.Base_Dir}/data"
    EnvParam.SpreadSheet_Dir = f"{EnvParam.Base_Dir}/output/{AlgParam.DataName}"
    EnvParam.MagInfoOutput_ON = True
    EnvParam.Test_ON = True
    EnvParam.Plot_ON = True
    return EnvParam

def print_structure(d, indent=0):
    """Recursively prints the structure of a dictionary."""
    for key, value in d.items():
        print(" " * indent + str(key) + ":", end=" ")
        if isinstance(value, dict):
            print()
            print_structure(value, indent + 2)
        elif isinstance(value, list):
            print("[", end="")
            print(", ".join(str(type(i)) for i in value), end="]\n")
        else:
            print(type(value))


def Spikelet_exec(D, AlgParam, EnvParam):
    DataFileName = (
        AlgParam.DataFileName
        if hasattr(AlgParam, "DataFileName")
        else AlgParam.DataName
    )
    #Thr_Str = "" #get_Thr_Str(AlgParam) es hat bei dieser methode einen fehler gegeben, methode brauche ich aber gerade eh nicht
    Thr_Str = get_Thr_Str(AlgParam)

    # Optional: Omit the following if no file output is needed
    MagInfo_File = EnvParam.get("MagInfo_File", f"MagInfo_{DataFileName}_{Thr_Str}.mat")
    MagInfoOutput_ON = EnvParam.get("MagInfoOutput_ON", False)
    Test_ON = EnvParam.get("Test_ON", False)
    CalcMp_ON = getattr(
        AlgParam, "CalcMp_ON", True
    )  # getattr() weil AlgParam keine .get() funktion hat. EnvParam ist anscheinend ein Dictionary
    Plot_ON = EnvParam.get("Plot_ON", False)

    if not CalcMp_ON:
        Plot_ON = False

    Param = Spikelet_MpParam_generate_ver_02(AlgParam)
    MagInfo = Spikelet_aproximation_ver_03(D, Param)

    return MagInfo

    file_path = r'C:\Users\Victor\Desktop\Uni\Bachelor\stuff\MagInfo_post_approx.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(MagInfo, f)
    print(f"Dictionary erfolgreich unter {file_path} gespeichert.")

    # Optional: Keep if you need to print or log symbols
    MagInfo["dataname"] = DataFileName
    MagInfo = Spikelet_Char_query(MagInfo)
    print_char_symbol(MagInfo)
    MagInfo, QueryRslt = Spikelet_Word_query(MagInfo, Param.query)
    print_word_symbol(MagInfo)

    if CalcMp_ON:
        MagInfo, MpRslt = Spikelet_MP_new_ver_02(MagInfo, Param)
        MagInfo, MpNnEach = Spikelet_MP_new_ver_02(MagInfo, Param)

    if Plot_ON:
        Spikelet_MP_plot_all(MagInfo, EnvParam)

    if MagInfoOutput_ON:
        MagInfo_Path = f"{EnvParam['SpreadSheet_Dir']}/{MagInfo_File}"
        np.save(MagInfo_Path, MagInfo)

    TestRslt = None

    if Test_ON:
        MagInfo_standard_Path = f"{EnvParam['SpreadSheet_Standard_Dir']}/{MagInfo_File}"
        MagInfo_standard = np.load(MagInfo_standard_Path, allow_pickle=True).item()
        TestRslt = MagInfo == MagInfo_standard

    return MagInfo, TestRslt, Param


def get_Thr_Str(AlgParam):
    M_Str = f"m{AlgParam.magnitude_threshold}" if hasattr(AlgParam, 'magnitude_threshold') else ""
    C_Str = f"c{AlgParam.constant_length_threshold}" if hasattr(AlgParam, 'constant_length_threshold') else ""
    if not M_Str and not C_Str:
        print("auto was chosen")
        return "auto"
    return f"{M_Str}_{C_Str}".replace('.', 'p')



def print_char_symbol(MagInfo):
    SpikeDb = MagInfo["spikeDb"]
    CMS_str = SpikeDb["alphabet"]
    for i, char in enumerate(CMS_str, start=1):
        if i % 5 == 1:
            from_idx = i
            print(f"{i}: ", end="")
        print(char, end="")
        if i % 5 == 0 or i == len(CMS_str):
            to_idx = i
            print(f" {from_idx}-{to_idx}")


def print_word_symbol(MagInfo):
    CMS_str = MagInfo["spikeDb"]["alphabet"]
    QR = MagInfo["word_query_result"]
    Names = QR["string_query_names"]
    for q_id, SQ in enumerate(QR["string_query"], start=1):
        pattern = QR["query"][q_id - 1]["pattern"]
        print(f"\n[qid = {q_id}] {pattern}")
        for row in SQ:
            local_id = row[Names == "local_id"]
            from_idx = row[Names == "from"]
            to_idx = row[Names == "to"]
            start_spike = row[Names == "start_spike_id"]
            string_length = row[Names == "string_length"]
            end_spike = start_spike + string_length - 1
            symbol_rep = CMS_str[start_spike - 1 : end_spike]
            print(f"[{local_id}] {symbol_rep} ({from_idx}:{to_idx})")


def Spikelet_MP_plot_all(MagInfo, EnvParam):
    pass

def motif_discovery_and_clasp(X):
    env_param = {}
    alg_param = AlgParam()

    # Set values for MaT and CoT
    #alg_param.magnitude_threshold = 0.5  # Example value for MaT
    #alg_param.constant_length_threshold = 25  # Example value for CoT

    MagInfo = Spikelet_exec(X, alg_param, env_param)

    transformed_data = MagInfo["data"]

    return transformed_data
