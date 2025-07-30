# GENDEROUS

We present GENDEROUS, a dataset of gender-ambiguous sentences containing gender-marked occupations and adjectives, and sentences with the ambiguous or non-binary pronoun ‘their’, as introduced in Hackenbuchner et al. (2025). We cross-linguistically evaluated how machine translation (MT) systems ([Google Translate](https://translate.google.com) and [DeepL](https://www.deepl.com/en/translator)) and large language models (LLMs) ([GPT-4o](https://chatgpt.com/) (``gpt-4o-2024-11-20``) and [EuroLLM-9B](https://huggingface.co/utter-project/EuroLLM-9B-Instruct)) translate these sentences from English into four grammatical gender languages: Greek, German, Spanish and Dutch. 

This repository includes the original English dataset (in ``raw_data``), the translation python scripts (in ``scripts`` both for LLMs and MT), the translations into the four respective target languages for each system (in ``translations``), and annotations for each set of translations (both human gold labels and GPT-4o annotations, in ``annotations``).

## Raw Data
Contains in separate files:
- base sentences
- sentences including adjectives
- sentences including `their`
- list of gender-inflected adjectives used to compile dataset
- list of stereotypical occupations used to compile dataset

## Translations
Contains in separate files:
- MT translations for all raw sentence files above for both DeepL and Google Translate into Greek, German, Spanish and Dutch
- LLM default translations (prompt1) for all raw sentence files for both EuroLLM and GPT-4o into Greek, German, Spanish and Dutch
- LLM translations for alternative genders (prompt2) for base sentence file for GPT-4o into Greek, German, Spanish and Dutch

## Annotations
Contains each of the above translated files with gender annotations both by human annotators (gold labels) and by GPT-4o (LLM-as-a-Judge).

## Scripts
Contains:
- Scripts to run translations for LLMs, both EuroLLM and GPT-4o: python script (``translate_llms.py``), shell script (``translate_llms.sh``), and utils (``utils_llms.py``)
- Scripts to run translations for MTs, both DeepL and Google Translate: python script (``translate_MT.py``) and shell script (``translate_MT.sh``)

The scripts contain information in comments about where you need to change which variables, including in shell script:
- information for computing centre
- directory
- HuggingFace Token or API Keys
- Choosing the model (either LLM or MT)
- Choosing the target language or prompt
- Defining input data (file/directory)
- in ``.py`` script: check your home directory


# Acknowledgements
This study has been partially funded by The Research Foundation – Flanders (FWO), research project 1SH5V24N (from 01.11.2023 until 31.10.2027), and hosted within the Language and Translation Technology Team (LT3) at Ghent University. The computational resources (Stevin Supercomputer Infrastructure) and services partially used in this work were provided by the VSC (Flemish Supercomputer Center), funded by Ghent University, FWO and the Flemish Government – department EWI.

# Cite
@inproceedings{hackenbuchner-etal-2025-genderous,
    title = "{GENDEROUS}: Machine Translation and Cross-Linguistic Evaluation of a Gender-Ambiguous Dataset",
    author = "Hackenbuchner, Jani{\c{c}}a  and
      Daems, Joke  and
      Gkovedarou, Eleni",
    editor = "Fale{\'n}ska, Agnieszka  and
      Basta, Christine  and
      Costa-juss{\`a}, Marta  and
      Sta{\'n}czak, Karolina  and
      Nozza, Debora",
    booktitle = "Proceedings of the 6th Workshop on Gender Bias in Natural Language Processing (GeBNLP)",
    month = aug,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.gebnlp-1.27/",
    pages = "302--319",
    ISBN = "979-8-89176-277-0"
}
