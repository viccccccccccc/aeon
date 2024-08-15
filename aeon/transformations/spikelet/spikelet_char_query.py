import numpy as np

def Spikelet_Char_query(MagInfo):
    func_name = 'Spikelet_Char_query'
    SuppMax = MagInfo['spikeDb']['symbol_mapping']['supp_max']
    SuppMin = MagInfo['spikeDb']['symbol_mapping']['supp_min']
    
    # Define the valid range of types
    ValidRange = [0, 1, 2, -1, -2, np.nan]

    # Extract time and magnitude information
    tm = MagInfo['output']['generateInitialSpikelet']['tm']
    tm = np.reshape(tm, (-1, 1))  # Ensure tm is a column vector
    mag = MagInfo['output']['generateInitialSpikelet']['mag']
    mag = np.reshape(mag, (-1, 1))  # Ensure mag is a column vector

    # Find indices within the valid range
    Type = MagInfo['output']['generateInitialSpikelet']['type']
    loc = np.isin(Type, ValidRange)
    
    # Default symbol mapping
    symbol_mapping_rule = MagInfo['spikeDb']['symbol_mapping']['rule']
    Symbol = np.full_like(Type, 'N', dtype='<U1')  # Initialize symbol array with 'N'

    # Set default character information
    spike_info = {
        'type': np.full_like(Type, np.nan, dtype=np.float64),
        'symbol': np.full_like(Type, '', dtype='<U1'),
        'type_supp_max': np.full_like(Type, np.nan, dtype=np.float64),
        'type_supp_0': np.full_like(Type, np.nan, dtype=np.float64),
        'type_supp_100': np.full_like(Type, np.nan, dtype=np.float64),
    }
    
    # Assign symbol for locations within valid range
    for rule in symbol_mapping_rule:
        condition = rule['condition']
        symbol = rule['symbol']
        mask = eval(condition)  # Apply condition to determine where the rule applies
        Symbol[loc & mask] = symbol

    # Set output
    MagInfo['output']['generateInitialSpikelet']['symbol'] = Symbol

    # Calculate support for each type
    for i in range(len(Type)):
        if np.isnan(Type[i]):
            continue
        current_type = Type[i]
        indices = (Type == current_type)
        if not np.any(indices):
            continue
        spike_info['type_supp_max'][i] = np.sum(indices)
        spike_info['type_supp_0'][i] = np.min(tm[indices])
        spike_info['type_supp_100'][i] = np.max(tm[indices])
    
    # Update MagInfo with spike information
    MagInfo['spikeDb']['character'] = spike_info

    # Display information about symbols
    for symbol in np.unique(Symbol):
        if symbol != 'N':
            symbol_indices = (Symbol == symbol)
            support_count = np.sum(symbol_indices)
            print(f"Symbol {symbol}: count = {support_count}")

    return MagInfo
