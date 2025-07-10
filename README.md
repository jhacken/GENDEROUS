# GENDEROUS

We present GENDEROUS, a dataset of gender-ambiguous sentences containing gender-marked occupations and adjectives, and sentences with the ambiguous or non-binary pronoun ‘their’, as introduced in Hackenbuchner et al. (2025). We cross-linguistically evaluated how machine translation (MT) systems ([Google Translate](https://translate.google.com) and [DeepL](https://www.deepl.com/en/translator)) and large language models (LLMs) ([GPT-4o](https://chatgpt.com/) (``gpt-4o-2024-11-20``) and [EuroLLM-9B](https://huggingface.co/utter-project/EuroLLM-9B-Instruct)) translate these sentences from English into four grammatical gender languages: Greek, German, Spanish and Dutch. 

<p align="center" width="100%">
  <img width="350" height="250" alt="image (1)" src="https://github.com/user-attachments/assets/752118f3-5292-48ff-baf7-0c61bcd7bf9c" />
</p>

This repository includes the original English dataset (in ``raw_data``), the translation python scripts, the translations into the four respective target languages for each system (in ``translations``), and annotations for each set of translations (both human gold labels and GPT-4o annotations, in ``annotations``).


# Acknowledgements
This study has been partially funded by The Research Foundation – Flanders (FWO), research project 1SH5V24N (from 01.11.2023 until 31.10.2027), and hosted within the Language and Translation Technology Team (LT3) at Ghent University. The computational resources (Stevin Supercomputer Infrastructure) and services partially used in this work were provided by the VSC (Flemish Supercomputer Center), funded by Ghent University, FWO and the Flemish Government – department EWI.

# References
Hackenbuchner, Janiça; Gkovedarou, Eleni; and Daems, Joke (2025): GENDEROUS: Machine Translation and Cross-Linguistic Evaluation of a Gender-Ambiguous Dataset. 
