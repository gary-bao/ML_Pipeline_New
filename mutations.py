# ml_pipeline/mutations.py
from typing import List
valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
residues = ['A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def is_valid(seq: str) -> bool:
    if not isinstance(seq, str):
        return False
    return set(seq.upper()).issubset(valid_aas)

def mutate(peptide: str) -> List[str]:
    non_cys_positions = [i for i, x in enumerate(peptide) if x != 'C']
    children = sorted(set([peptide[:pos] + residue + peptide[pos+1:] for pos in non_cys_positions for residue in residues]))
    # keep the parent also
    if peptide not in children:
        children.append(peptide)
        children.sort()
    return children

def mutate_double(peptide: str) -> List[str]:
    children = []
    for child in mutate(peptide):
        children += mutate(child)
    children = sorted(set(children))
    if peptide not in children:
        children.append(peptide)
        children.sort()
    return children
