from pathlib import Path

import h5py
import numpy as np


def load_data_ASCAD(filename, target_byte=None):

    file = h5py.File(str(filename)+".h5")

    profile = file["Profiling_traces"]
    attack = file["Attack_traces"]

    profiling_traces = np.array(profile["traces"])
    profiling_plaintext = profile["metadata"]["plaintext"]
    profiling_key = profile["metadata"]["key"]

    attack_traces = np.array(attack["traces"])
    attack_plaintext = attack["metadata"]["plaintext"]
    attack_key = attack["metadata"]["key"]

    if target_byte:
        profiling_plaintext = profiling_plaintext[:, target_byte]
        profiling_key = profiling_key[:, target_byte]
        attack_plaintext = attack_plaintext[:, target_byte]
        attack_key = attack_key[:, target_byte]
        
    #!!!!!!using smaller dataset    
    profilingTracesShort = profiling_traces[:5000]
    profilingPlaintextShort = profiling_plaintext[:5000]
    profilingKeyShort = profiling_key[:5000]
    
    attackTracesShort = attack_traces[:1000]
    attackPlaintextShort = attack_plaintext[:1000]
    attackKeyShort = attack_key[:1000]

#!!!!!!uncomment if not using smaller dataset!!!!!!
#    return (
#        (profiling_traces, profiling_plaintext, profiling_key),
#        (attack_traces, attack_plaintext, attack_key)
#    )
    
#!!!!!!comment if not using smaller dataset!!!!!!
    return (
        (profilingTracesShort, profilingPlaintextShort, profilingKeyShort),
        (attackTracesShort, attackPlaintextShort, attackKeyShort)
    )


def load_data_ches_ctf(filename):

    file = h5py.File(str(filename)+".h5")

    profiling_traces = np.array(file.get('profiling_traces'))
    profiling_data = np.array(file.get('profiling_data'))
    attack_traces = np.array(file.get('attacking_traces'))
    attack_data = np.array(file.get('attacking_data'))

    (profiling_plaintext,
     profiling_output,
     profiling_key) = np.split(profiling_data, [16, 32], axis=1)

    (attack_plaintext,
     attack_output,
     attack_key) = np.split(attack_data, [16, 32], axis=1)

    return (
        (profiling_traces, profiling_plaintext, profiling_key),
        (attack_traces, attack_plaintext, attack_key)
    )


def load_data_chipwhisperer(data_root):

    data_root = Path(data_root)

    traces = np.load(data_root/"traces.npy")
    pt = np.load(data_root/"plain.npy")
    key = np.load(data_root/"key.npy")

    train_slice = slice(0, 2000)
    test_slice = slice(9990, 10000)

    profiling_traces = traces[train_slice]
    profiling_plaintext = pt[train_slice]
    profiling_key = key[train_slice]

    attack_traces = traces[test_slice]
    attack_plaintext = pt[test_slice]
    attack_key = key[test_slice]

    return (
        (profiling_traces, profiling_plaintext, profiling_key),
        (attack_traces, attack_plaintext, attack_key)
    )


string_contains_to_loader = {
    "ascad": load_data_ASCAD,
    "ches_ctf": load_data_ches_ctf,
    "chipwhisperer": load_data_chipwhisperer
}


def select_traget_bype(data, target_byte):

    ((profiling_traces, profiling_plaintext, profiling_key),
     (attack_traces, attack_plaintext, attack_key)) = data

    profiling_plaintext = profiling_plaintext[:, target_byte]
    profiling_key = profiling_key[:, target_byte]
    attack_plaintext = attack_plaintext[:, target_byte]
    attack_key = attack_key[:, target_byte]

    return (
        (profiling_traces, profiling_plaintext, profiling_key),
        (attack_traces, attack_plaintext, attack_key)
    )


def load_data(path, target_byte=None):

    data = None
    for string, loader in string_contains_to_loader.items():
        if string in str(path):
            data = loader(path)
            if target_byte is not None:
                data = select_traget_bype(data, target_byte)
            return data

    raise RuntimeError(
        f"Unrecognized dataset, currently supported are {string_contains_to_loader.keys()}"
    )
