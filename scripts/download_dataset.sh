# Download Training and Validation dataset
# This curated dataset is preprocessed from Plinder
wget https://zenodo.org/records/15178859/files/train_val.zip
wget https://zenodo.org/records/15220255/files/train_val_samples_weights.json

# Download MSA Features
wget https://zenodo.org/records/15178859/files/msa_features_aa
wget https://zenodo.org/records/15178859/files/msa_features_ab
wget https://zenodo.org/records/15178859/files/msa_features_ac
wget https://zenodo.org/records/15206943/files/msa_features_ad
wegt https://zenodo.org/records/15206943/files/msa_features_ae

cat msa_features_aa msa_features_ab msa_features_ac msa_features_ad msa_features_ae > msa_features.tar.gz
tar -zxvf msa_features.tar.gz
# Download Uniprot MSA Features
wget https://zenodo.org/records/15206943/files/uniprot_msa_features_aa
wget https://zenodo.org/records/15206943/files/uniprot_msa_features_ab
wget https://zenodo.org/records/15209515/files/uniprot_msa_features_ac
wget https://zenodo.org/records/15209515/files/uniprot_msa_features_ad
wget https://zenodo.org/records/15209515/files/uniprot_msa_features_ae
wget https://zenodo.org/records/15209515/files/uniprot_msa_features_af
wget https://zenodo.org/records/15210625/files/uniprot_msa_features_ag
wget https://zenodo.org/records/15210625/files/uniprot_msa_features_ah

cat uniprot_msa_features_aa uniprot_msa_features_ab uniprot_msa_features_ac \
  uniprot_msa_features_ad uniprot_msa_features_ae uniprot_msa_features_af \
  uniprot_msa_features_ag uniprot_msa_features_ah > uniprot_msa_features.tar.gz
tar -zxvf uniprot_msa_features.tar.gz