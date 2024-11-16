from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem

def read_csv_to_df(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    return df


def normalize_df(data):
    # Create a StandardScaler object
    scaler = StandardScaler()
    
    if isinstance(data, pd.Series):
        # If the input is a Series, convert it to a DataFrame first, normalize, and convert back to Series
        data = pd.DataFrame(data)
        normalized_data = scaler.fit_transform(data)
        return pd.Series(normalized_data.flatten(), index=data.index)
    elif isinstance(data, pd.DataFrame):
        # Identify numeric columns in the DataFrame
        numeric_columns = data.select_dtypes(include=['number']).columns
        
        # Fit and transform the numeric columns
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        
        return data
    else:
        raise ValueError("Input should be a pandas DataFrame or Series.")

def canonicalize_list(smiles_list):
    canonical_smiles_list = []
    mol_list = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:  # Only process valid molecules
                canon = Chem.MolToSmiles(mol, isomericSmiles=True)
                canonical_smiles_list.append(canon)
                mol_list.append(mol)
        except Exception:
            continue  # Skip invalid SMILES without adding None
    
    return canonical_smiles_list, mol_list



def get_descriptors(mol_list, use_mordred=False):
    
    # Initialize Mordred calculator only if needed
    calc = Calculator(descriptors, ignore_3D=True) if use_mordred else None

    # List of RDKit descriptor functions
    rdkit_descriptors = {
        "Molecular Weight": Descriptors.MolWt,
        "LogP": Descriptors.MolLogP,
        "TPSA": Descriptors.TPSA,
        "Number of Rings": Chem.rdMolDescriptors.CalcNumRings,
        "H-Bond Donors": Descriptors.NumHDonors,
        "H-Bond Acceptors": Descriptors.NumHAcceptors,
        "Rotatable Bonds": Descriptors.NumRotatableBonds,
        "Heavy Atom Count": Descriptors.HeavyAtomCount,
        "Aromatic Ring Count": Chem.rdMolDescriptors.CalcNumAromaticRings,
        "Fraction Csp3": Descriptors.FractionCSP3,
        "Number of Heteroatoms": Descriptors.NumHeteroatoms,
        "Number of Aliphatic Rings": Chem.rdMolDescriptors.CalcNumAliphaticRings,
        "Number of Aliphatic Carbons": Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles,
        "Number of Saturated Rings": Chem.rdMolDescriptors.CalcNumSaturatedRings,
        "Number of Aromatic Heterocycles": Chem.rdMolDescriptors.CalcNumAromaticHeterocycles,
        "Molecular Refractivity": Descriptors.MolMR
    }


    data = []

    for mol in mol_list:
        # Compute RDKit descriptors
        desc = {name: func(mol) for name, func in rdkit_descriptors.items()}

        # Compute Mordred descriptors if requested
        if use_mordred and calc:
            mordred_desc = calc(mol)
            desc.update({
                "Total Atom Count": mordred_desc["nAtom"],
                "Balaban Index": mordred_desc["BalabanJ"],
                "Petitjean Index": mordred_desc["PetitjeanIndex"]
            })

        data.append(desc)

    # Return the data as a DataFrame
    return pd.DataFrame(data)
