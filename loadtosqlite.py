import pandas as pd
from sqlalchemy import create_engine

# Database connection
engine = create_engine('sqlite:///gdsc_data.db')  # SQLite database, adjust as needed for other DBs
#/Users/DipalShah/Downloads/pubchem192drugs.csv
# Load CSV files
gene_expression_df = pd.read_csv('/Users/DipalShah/Downloads/AllGeneExpression.csv')
drug_description_df = pd.read_csv('/Users/DipalShah/Downloads/Drug_Desciption_GDSC.csv')
dose_response_df = pd.read_csv('/Users/DipalShah/Downloads/GDSC2_fitted_dose_response_25Feb20_3.csv')
pubchem_details_df = pd.read_csv('/Users/DipalShah/Downloads/pubchem192drugs.csv')

# Melt the GeneExpression DataFrame to convert columns into rows
gene_expression_melted = gene_expression_df.melt(id_vars=["SAMPLE_ID"],
                                                 var_name="Gene",
                                                 value_name="ExpressionValue")

# Rename columns in pubchem_details_df to match the relational schema
pubchem_details_df.rename(columns={
    'DRUG_NAME': 'DrugName',
    'PUBCHEM_ID': 'PubChemID',
    'MOLECULAR_FORMULA': 'MolecularFormula',
    'MOLECULAR_WEIGHT': 'MolecularWeight',
    'CANONICAL_SMILES': 'CanonicalSMILES',
    'ISOMERIC_SMILES': 'IsomericSMILES',
    'INCHI': 'InChI',
    'INCHIKEY': 'InChIKey',
    'IUPAC_NAME': 'IUPACName',
    'XLOGP': 'XLogP',
    'SYNONYMS': 'Synonyms'
}, inplace=True)

# Save dataframes to SQL database
gene_expression_melted.to_sql('GeneExpression', con=engine, if_exists='replace', index=False)
drug_description_df.to_sql('DrugDescription', con=engine, if_exists='replace', index=False)
dose_response_df.to_sql('DoseResponse', con=engine, if_exists='replace', index=False)
pubchem_details_df.to_sql('DrugPubChem', con=engine, if_exists='replace', index=False)

print("Data successfully loaded into the database.")
