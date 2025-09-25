# ml_pipeline/features.py
"""
Featurization helpers:
- SMILES builder from peptide sequence
- Morgan fingerprint extraction (numpy array)
"""

from typing import List
import logging
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit import DataStructs

# Units map 
units = {
    'A': 'N[C@@H](C)C(=O)-',
    'C': 'N[C@@H](CS*)C(=O)-',
    'D': 'N[C@@H](CC(=O)[O-]*)C(=O)-',
    'E': 'N[C@@H](CCC(=O)[O-]*)C(=O)-',
    'F': 'N[C@@H](Cc%ccccc%)C(=O)-',
    'G': 'NCC(=O)-',
    'H': 'N[C@@H](Cc%cnc[nH]%)C(=O)-',
    'I': 'N[C@@H]([C@@H](C)CC)C(=O)-',
    'K': 'N[C@@H](CCCC[NH3+])C(=O)-',
    'L': 'N[C@@H](CC(C)C)C(=O)-',
    'M': 'N[C@@H](CCSC)C(=O)-',
    'N': 'N[C@@H](CC(=O)N)C(=O)-',
    'P': 'N%[C@@H](CCC%)C(=O)-',
    'Q': 'N[C@@H](CCC(=O)N)C(=O)-',
    'R': 'N[C@@H](CCCNC(N)=[NH2+])C(=O)-',
    'S': 'N[C@@H](CO)C(=O)-',
    'T': 'N[C@@H]([C@H](O)C)C(=O)-',
    'V': 'N[C@@H](C(C)C)C(=O)-',
    'W': 'N[C@@H](Cc%c[nH]c&ccccc%&)C(=O)-',
    'Y': 'N[C@@H](Cc%ccc(O)cc%)C(=O)-',
    'NH2': 'N'
}


def get_parts(peptide: str) -> List[str]:
    parts = list(peptide)
    cys_locations = [i for i, x in enumerate(parts) if x == 'C']
    try:
        first_cys = min(cys_locations)
    except ValueError:
        first_cys = -1
    try:
        last_cys = max(cys_locations)
    except ValueError:
        last_cys = -1
    if 0 <= first_cys < last_cys:
        parts[first_cys] += '*'
        parts[last_cys] += '*'
    parts.append('NH2')
    return parts


def get_smiles(peptide: str) -> str:
    """
    Convert peptide sequence (e.g. 'ACD...') to concatenated SMILES string using `units` blocks.
    """
    smiles = ''
    cycles = {}
    ring_number = 1
    for part in get_parts(peptide):
        unit = part.rstrip('*!')
        cycle_labels = part[len(unit):]
        if unit not in units:
            logging.warning(f"Unknown unit '{unit}' in peptide '{peptide}' — skipping unit.")
            continue
        block = units[unit].replace('%', str(ring_number)).replace('&', str(ring_number+1))
        if cycle_labels:
            for character in cycle_labels:
                if character in cycles:
                    ring_number -= 1
                    block = block.replace('*', str(ring_number)).replace(character, str(ring_number))
                else:
                    block = block.replace('*', str(ring_number)).replace(character, str(ring_number))
                    cycles[character] = ring_number
                    ring_number += 1
        block = block.replace('*', '')
        smiles += block
    return smiles


def featurize(seq: str, fp_size: int = 2048, radius: int = 3) -> np.ndarray:
    """
    Convert peptide sequence -> SMILES -> Morgan fingerprint (numpy array of 0/1 ints).
    Returns numpy array of shape (fp_size,).
    """
    if not isinstance(seq, str):
        return np.zeros(fp_size, dtype=np.int8)

    smiles = get_smiles(seq)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.info(f"RDKit could not parse SMILES for sequence {seq}")
        return np.zeros(fp_size, dtype=np.int8)

    try:
        fpgen = AllChem.GetMorganGenerator(fpSize=fp_size, radius=radius)
        fp_array = fpgen.GetFingerprintAsNumPy(mol)
        return fp_array
    except Exception as e:
        logging.exception(f"Error computing fingerprint for {seq}: {e}")
        return np.zeros(fp_size, dtype=np.int8)
