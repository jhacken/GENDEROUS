#!/bin/bash -l
#PBS -l nodes=1:ppn=24:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=00:20:00
#PBS -N LLMs
#PBS -m abe

echo "Loading modules"

ml purge
ml load Python/3.12.3-GCCcore-13.3.0

source ./venv/bin/activate                                                  ## Check your venv

pip show httpx openai
pip install accelerate
pip install python-dotenv
pip install tenacity
pip install tyro
pip install inflect
pip install vllm

echo "Current Python interpreter:"
which python

# Change to the directory where the Python script is located                ## Check your directory
cd $PBS_O_WORKDIR

# Check CUDA version
echo "Checking CUDA version for PyTorch and Environment:"
python -c "import torch; print(torch.version.cuda)"

# Check GPU availability
echo "Checking GPU availability:"
nvidia-smi


## Setting up API Keys
echo "Accessing API Keys (OpenAI/HF Token)."
export HF_TOKEN="INSERT-HF-TOKEN-HERE"                                       ## Add your HuggingFace token
#export OPENAI_API_KEY="INSERT_KEY"                                          ## Add your OpenAI API key

## Uncomment for EuroLLM HF access
huggingface-cli login --token $HF_TOKEN


echo "Starting the Python script..."

prompt_template="prompt1_NL"                                                ## Choose your prompt per language.

## CHOOSE YOUR MODEL                                                        ## Choose your model (GPT or EuroLLM)
## GPT-4o
#model="gpt-4o-2024-11-20"+

## EuroLLM
model="utter-project/EuroLLM-9B-Instruct"


# Set output file based on model
if [[ "$model" == *"gpt"* ]]; then
    output_file="./results/gpt_${prompt_template}.tsv"
else
    # For EuroLLM
    model_safe=$(echo "$model" | sed 's/\//_/g')
    output_file="./results/${model_safe}_${prompt_template}.tsv"
fi

# Create results directory if it doesn't exist
mkdir -p ./results

# Check if input dataset exists
dataset_file="./data/30_base_sentences_occupations.tsv"                     ## Define your input data. Files: base_sentences, their_sentences, adjective_sentences

echo "Model: $model"
echo "Output file will be: $output_file"
echo "Dataset file: $dataset_file"


python translate_llms.py \
    --dataset_file $dataset_file \
    --output_file $output_file \
    --model_name_or_path $model \
    --prompt_template $prompt_template


echo "Python script has finished running."