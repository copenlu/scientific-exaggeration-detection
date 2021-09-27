# Semi-Supervised Exaggeration Detection of Health Science Press Releases

Dustin Wright and Isabelle Augenstein

In EMNLP 2021

https://arxiv.org/pdf/2108.13493.pdf%20

*Code is currently being prepared! Curated test data is now available.*

<p align="center">
  <img src="exaggeration.png" alt="Exaggeration Detection">
</p>

Public trust in science depends on honest and factual communication of scientific papers. However, recent studies have demonstrated a tendency of news media to misrepresent scientific papers by exaggerating their findings. Given this, we present a formalization of and study into the problem of exaggeration detection in science communication. While there are an abundance of scientific papers and popular media articles written about them, very rarely do the articles include a direct link to the original paper, making data collection challenging. We address this by curating a set of labeled press release/abstract pairs from existing expert annotated studies on exaggeration in press releases of scientific papers suitable for benchmarking the performance of machine learning models on the task. Using limited data from this and previous studies on exaggeration detection in science, we introduce MT-PET, a multi-task version of Pattern Exploiting Training (PET), which leverages knowledge from complementary cloze-style QA tasks to improve few-shot learning. We demonstrate that MT-PET outperforms PET and supervised learning both when data is limited, as well as when there is an abundance of data for the main task.

## Data

The training and test data derived from the studies from [Sumner et al. 2014](https://www.bmj.com/content/349/bmj.g7015) and [Bratton et al. 2019](https://pubmed.ncbi.nlm.nih.gov/31728413/#:~:text=Results%3A%20We%20found%20that%20the,inference%20from%20non%2Dhuman%20studies.) can be found in the `data` directory. Files are formatted as json lines files, with each row containing the following fields:

```
original_file_id: The ID of the original spreadsheet in the Sumner/Bratton data where the annotations are derived from
press_release_conclusion: The conclusion sentence from the press release
press_release_strength: The strength label for the press release
abstract_conclusion: The conclusion sentence from the abstract
abstract_strength: The strength label for the abstract
exaggeration_label: The final exaggeration label
```

The exaggeration label is one of `same`, `exaggerates`, or `downplays`. The strength label is one of the following:

```
0: Statement of no relationship
1: Statement of correlation
2: Conditional statement of causation
3: Statement of causation
```

We used the data from `insciout_test.jsonl` as test data in all of the experiments in the paper. The claim strength data from [Yu et al. 2019](https://aclanthology.org/D19-1473/) and [Yu et al. 2020](https://aclanthology.org/2020.coling-main.427/) can be found [here](https://github.com/junwang4/correlation-to-causation-exaggeration/tree/master/data). We will release the unsupervised data (and/or the scripts needed to collect unsupervised press release/abstract pairs) shortly.

## MT-PET

The code to run multi-task PET is included as a submodule pointing to [our PET fork](https://github.com/dwright37/mt-pet). Either clone that repository or pull the submodule by running `git submodule update --init --recursive`

## Citation

```
@inproceedings{wright2021exaggeration,
    title={{Semi-Supervised Exaggeration Detection of Health Science Press Releases}},
    author={Dustin Wright and Isabelle Augenstein},
    booktitle = {Proceedings of EMNLP},
    publisher = {Association for Computational Linguistics},
    year = 2021
}
```
