#!/bin/bash -l
#PBS -l nodes=1:ppn=24:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=00:20:00
#PBS -N MT
#PBS -m abe

echo "Loading modules"

ml purge
ml load Python/3.12.3-GCCcore-13.3.0

source venv/bin/activate                                        ## Activate your virtual environment

pip install accelerate
pip install python-dotenv
pip install google-cloud-translate==2.0.1
pip install tyro

echo "Current Python interpreter:"
which python

# Change to the directory where the Python script is located       ## Check your directory
cd $PBS_O_WORKDIR

# Check CUDA version
echo "Checking CUDA version for PyTorch and Environment:"
python -c "import torch; print(torch.version.cuda)"


## Setting up Google/DeepL API Key
export auth_key=""                                                 ## Add your Google or DeepL API Key

echo "Starting the Python script..."

## Models: DeepL or Google Translate                                ## Choose your MT model
#model="google-translate"
model="deepl"

## Target languages: DE, NL, ES, EL                                 ## Choose your target languages
target_language="ES"

# save output file as
output_file="./results/${model}_${target_language}.tsv"

# Create results directory if it doesn't exist
mkdir -p ./results

# Input dataset
dataset_file="./data/30_base_sentences_occupations.tsv"             ## Define your input data. Files: base_sentences, their_sentences, adjective_sentences

echo "Model: $model"
echo "Output file will be: $output_file"
echo "Dataset file: $dataset_file"

# Check if dataset file exists
if [[ ! -f "$dataset_file" ]]; then
    echo "Error: Dataset file '$dataset_file' not found!"
    exit 1
fi

python translate_MT_local_v3.py \
    --dataset_file "$dataset_file" \
    --output_file "$output_file" \
    --model_name_or_path "$model" \
    --target_language "$target_language" \
    --auth-key "$auth_key"

echo "Python script has finished running."