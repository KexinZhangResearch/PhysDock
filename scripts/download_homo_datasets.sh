# Set Donwload dir
DOWNLOAD_DIR=$1

mkdir -p $DOWNLOAD_DIR

# Download Uniref90
wget -P $DOWNLOAD_DIR/ https://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
gunzip $DOWNLOAD_DIR/uniref90.fasta.gz

# Download Mgnify
wget -P $DOWNLOAD_DIR/ https://storage.googleapis.com/alphafold-databases/v2.3/mgy_clusters_2022_05.fa.gz
gunzip $DOWNLOAD_DIR/mgy_clusters_2022_05.fa.gz

# Download Uniprot
wget -P $DOWNLOAD_DIR/ https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz
wget -P $DOWNLOAD_DIR/ https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz

gunzip $DOWNLOAD_DIR/uniprot_trembl.fasta.gz
gunzip $DOWNLOAD_DIR/uniprot_sprot.fasta.gz
  # Concatenate TrEMBL and SwissProt, rename to uniprot and clean up.
cat $DOWNLOAD_DIR/uniprot_sprot.fasta >> $DOWNLOAD_DIR/uniprot_trembl.fasta
mv $DOWNLOAD_DIR/uniprot_trembl.fasta $DOWNLOAD_DIR/uniprot.fasta
rm $DOWNLOAD_DIR/uniprot_sprot.fasta

# Download Uniclust30
wget -P $DOWNLOAD_DIR/ http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz
UNICLUST_DIR="${DOWNLOAD_DIR}/uniclust30"
mkdir -p "${UNICLUST_DIR}"
tar --extract --verbose --file="${DOWNLOAD_DIR}/uniclust30_2018_08_hhsuite.tar.gz" \
  --directory="${UNICLUST_DIR}"
rm "${DOWNLOAD_DIR}/uniclust30_2018_08_hhsuite.tar.gz"

# Download BFD
wget -P $DOWNLOAD_DIR/ https://storage.googleapis.com/alphafold-databases/casp14_versions/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
BFD_DIR="${DOWNLOAD_DIR}/bfd"
mkdir -p "${BFD_DIR}"
tar --extract --verbose --file="${DOWNLOAD_DIR}/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz" \
  --directory="${BFD_DIR}"
rm "${DOWNLOAD_DIR}/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz"
