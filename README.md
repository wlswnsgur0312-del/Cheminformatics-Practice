# 🧬 SMILES to 3D PDB Converter

### 📝 Project Overview
This project is a **Universal Cheminformatics Tool** designed to convert 1D chemical representations (SMILES) into optimized 3D molecular structures.
Regardless of the molecule's complexity, this script utilizes **RDKit** to generate 3D conformers and performs energy minimization (MMFF94) to ensure chemically valid geometries.
(SMILES 문자열만 있으면 어떤 분자든 에너지 최적화가 완료된 3D 구조(PDB)로 자동 변환해 주는 범용 도구입니다.)

### 🚀 Key Features
* **Universal Applicability:** Works with **ANY** valid SMILES string (e.g., Aspirin, Caffeine, Tetrodotoxin).
* **Automated Pipeline:** `SMILES` → `2D Topology` → `3D Embedding` → `Energy Optimization`.
* **Geometry Optimization:** Applies **MMFF94 (Merck Molecular Force Field)** to correct bond angles and minimize steric hindrance.
* **Standard Output:** Exports `.pdb` files compatible with PyMOL, Chimera, and other modeling software.

### 💻 Core Logic (Python)
The script defines a reusable function to process any given molecule.

```python
from rdkit import Chem
from rdkit.Chem import AllChem

def save_3d_structure(mol_name, smiles_string):
    """
    Converts a SMILES string to a 3D PDB file with MMFF optimization.
    """
    # 1. Convert SMILES to Mol object
    mol = Chem.MolFromSmiles(smiles_string)
    
    # 2. Add Hydrogens (Essential for 3D geometry)
    mol_3d = Chem.AddHs(mol)
    
    # 3. Generate 3D Conformer & Optimize
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_3d) # Force Field Optimization
    
    # 4. Save to PDB
    output_filename = f"{mol_name}.pdb"
    Chem.MolToPDBFile(mol_3d, output_filename)
    print(f">> Successfully saved '{output_filename}'")

# ==========================================
# ✅ Usage Example: Works for ANY molecule!

# Example 1: Aspirin
save_3d_structure("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O")

# Example 2: Tetrodotoxin (Complex Structure)
save_3d_structure("tetrodotoxin", "C(C1(C2C3C(N=C(NC34C(C1OC(C4O)(O2)O)O)N)O)O)O")

'''

<img width="896" height="919" alt="image" src="https://github.com/user-attachments/assets/13dbfdf2-7ef5-4ed7-ad08-97ac9e125bff" />



