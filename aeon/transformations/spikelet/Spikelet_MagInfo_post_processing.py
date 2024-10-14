import numpy as np

def Spikelet_MagInfo_post_processing(MagInfo, OpName):
    """
    Post-processing function for Spikelet decomposition.

    Parameters
    ----------
    MagInfo (dict): A dictionary containing the Spikelet decomposition information.
    OpName (str): The operation name to be used for output.

    Returns
    -------
    dict: Updated MagInfo dictionary after post-processing.
    """
    if "output" in MagInfo:
        # Neu berechnete Felder, abhängig von der aktuellen Operation
        MagInfo["center"] = np.where(~np.isnan(MagInfo["type"]))[0]
        MagInfo["data"] = Spikelet_get_TimeSeriesForm_from_SpikeletDecomposition(MagInfo)

        # Unabhängige Felder
        MagInfo["output"][OpName] = {}
        MagInfo["output"][OpName]["type"] = MagInfo["type"]
        MagInfo["output"][OpName]["value"] = MagInfo["value"]
        MagInfo["output"][OpName]["magnitude"] = MagInfo["magnitude"]
        MagInfo["output"][OpName]["leg_magnitude"] = MagInfo["leg_magnitude"]
        MagInfo["output"][OpName]["left"] = MagInfo["left"]
        MagInfo["output"][OpName]["right"] = MagInfo["right"]

        # Abhängige Felder
        MagInfo["output"][OpName]["data"] = MagInfo["data"]
        MagInfo["output"][OpName]["center"] = MagInfo["center"]

    else:
        # Falls `output` nicht vorhanden, speichere in `operation`
        MagInfo["operation"][OpName] = {}
        MagInfo["operation"][OpName]["data"] = MagInfo["data"]
        MagInfo["operation"][OpName]["magnitude"] = MagInfo["magnitude"]
        MagInfo["operation"][OpName]["left"] = MagInfo["left"]
        MagInfo["operation"][OpName]["right"] = MagInfo["right"]

    return MagInfo

def Spikelet_get_TimeSeriesForm_from_SpikeletDecomposition(MagInfo):
    """
    Generate the time series form from Spikelet decomposition.

    Parameters
    ----------
    MagInfo (dict): A dictionary containing the Spikelet decomposition information.

    Returns
    -------
    np.ndarray: The reconstructed time series data.
    """
    Data_org = MagInfo["data_org"]
    Center = np.where(~np.isnan(MagInfo["type"]))[0]
    Left = MagInfo["left"][Center]
    Right = MagInfo["right"][Center]

    Apx_time = np.unique(
        np.concatenate(([0], Left, Center, Right, [len(Data_org) - 1]))
    )

    if len(Apx_time) >= 2:
        Data = np.interp(np.arange(len(Data_org)), Apx_time, Data_org[Apx_time])
    else:
        Data = Data_org.copy()

    Center_const = np.where(MagInfo["type"] == 0)[0]

    for i in Center_const:
        from_idx = MagInfo["left"][i]
        to_idx = MagInfo["right"][i]
        value = MagInfo["value"][i]
        Data[from_idx : to_idx + 1] = value

    return Data