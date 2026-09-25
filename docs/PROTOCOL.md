# Experimental Protocol

## Project

**Working title:**  
Cross-Lingual Consistency of Counterfactual Financial Reasoning in Multilingual LLMs

## 1. Research Question

Does lexical/tokenizer overlap between languages predict how consistently a multilingual language model responds to controlled counterfactual changes in financial information?

## 2. Motivation

Qi, Fernández, and Bisazza (EMNLP 2023) found that lexical overlap is strongly associated with cross-lingual consistency in factual knowledge retrieval.

This project tests whether a similar relationship appears in a different setting: financial reasoning under controlled counterfactual interventions.

Instead of asking whether a model retrieves the same fact across languages, we test whether the model changes its financial judgment in the same way across languages when one economically meaningful fact is changed.

## 3. Hypotheses

### H1

Cross-lingual consistency in the effect of controlled counterfactual financial information will be positively associated with lexical/tokenizer overlap between language pairs.

Equivalently:

Higher tokenizer overlap should be associated with lower disagreement in the effect of the counterfactual intervention.

### H0

Lexical/tokenizer overlap between language pairs is not systematically associated with cross-lingual consistency in responses to counterfactual financial information.

## 4. Experimental Design

The primary experiment will use approximately 60 controlled financial minimal pairs.

Each item contains:

- an original financial statement
- a counterfactual version
- one controlled economically meaningful change
- an expected direction of market reaction

Examples of intervention families include:

1. profit direction
2. loss direction
3. cost direction
4. revenue direction
5. earnings beat versus miss
6. guidance raised versus lowered
7. margin expansion versus contraction
8. business or contract realization

The current eight DEV items are used only for software development and pipeline validation.

They are not part of the final experimental sample.

## 5. External Datasets

Two existing financial datasets will be used.

### Dataset A: Financial-news source corpus

A large recent financial-news/headline dataset will provide realistic financial events and language structures.

Preference will be given to information occurring after the documented training cutoff of the selected language model in order to reduce contamination from parametric memory.

Exact dataset name, release, size, date coverage, and language coverage must be verified from the original source before the dataset is frozen.

### Dataset B: Multilingual financial benchmark

A large multilingual financial classification/sentiment dataset will be used as a secondary external evaluation.

Its purpose is to check whether the model exhibits reasonable multilingual financial understanding outside the controlled counterfactual experiment.

It is not the primary hypothesis-testing dataset.

Exact dataset name, release, size, and language coverage must be verified from the original source before it is frozen.

## 6. Languages

Target: approximately seven languages covering substantially different degrees of lexical/tokenizer overlap.

The provisional set is:

- English
- Spanish
- German
- French
- Chinese
- Japanese
- Arabic

The final language set will be frozen after verifying:

1. coverage in the external datasets
2. support by the selected model
3. tokenizer behavior
4. feasibility of producing semantically equivalent translations

Languages must not be selected after observing the main experimental results.

## 7. Model

One pretrained multilingual instruction-following language model will be used for the primary experiment.

The model will be used for inference only.

No fine-tuning or training will be performed.

The exact model and model version must be frozen before the main experimental run.

The model tokenizer used for inference will also be used to calculate tokenizer overlap.

## 8. Model Output

For every financial statement, the model will predict the expected short-term effect on the company's stock price using exactly one ordinal label:

- -2 = strongly negative
- -1 = negative
-  0 = neutral or unclear
- +1 = positive
- +2 = strongly positive

The primary inference prompt will request only the label.

Generation settings will be deterministic or as close to deterministic as the selected model permits.

## 9. Counterfactual Effect

For item i and language l:

Delta(i,l) = y_counterfactual(i,l) - y_original(i,l)

This measures how much the model's financial judgment changes after the controlled counterfactual intervention.

The main object of interest is not whether every individual prediction is correct.

The main object of interest is whether the intervention produces a similar change across languages.

## 10. Cross-Lingual Disagreement

For languages a and b:

D(i,a,b) = |Delta(i,a) - Delta(i,b)|

Interpretation:

- smaller D = greater cross-lingual consistency
- larger D = greater cross-lingual disagreement

The analysis will retain observations at the item × language-pair level rather than reducing the experiment immediately to one aggregate value per language pair.

## 11. Tokenizer Overlap

Lexical similarity will be operationalized using the tokenizer of the selected multilingual model.

Overlap will be calculated from token sets observed in equivalent multilingual financial material.

Possible measures include:

- Jaccard overlap
- overlap coefficient

The exact primary overlap measure will be frozen before inspection of the main hypothesis-test results.

Language-family similarity alone will not be used as the primary lexical-overlap measure.

## 12. Statistical Analysis

The primary analysis will test whether greater tokenizer overlap is associated with lower cross-lingual disagreement.

Because language pairs share languages, the 21 possible pairs from seven languages cannot be treated as fully independent observations.

The final statistical procedure will therefore preserve item-level observations and account for repeated languages.

Candidate approaches include:

- Spearman association with a dependence-aware permutation procedure
- bootstrap inference with resampling at the financial-item level
- mixed-effects regression

One primary procedure will be selected and documented before the final hypothesis test.

Effect sizes and uncertainty will be reported in addition to statistical significance.

A null result is considered a valid research result.

## 13. Translation Control

Translations must preserve the financial semantics of each original/counterfactual pair.

The counterfactual intervention must remain the only economically meaningful difference between the two conditions.

Translations will be validated before the main inference run.

The translation procedure and any manual corrections will be documented.

## 14. Contamination Control

The final experiment should minimize the possibility that results are driven by memorized historical news.

Where real financial events are used, preference will be given to events occurring after the selected model's documented knowledge/training cutoff.

Controlled synthetic or anonymized statements may also be used where this produces a cleaner intervention.

## 15. Scope

The primary project consists of:

1. external financial data preparation
2. controlled counterfactual construction
3. multilingual translation
4. multilingual LLM inference
5. tokenizer-overlap measurement
6. cross-lingual consistency analysis
7. hypothesis testing

RAG and explicit retrieval-based knowledge conflicts are not part of the primary experiment.

They may be considered only as an optional extension if the primary experiment is completed first.

## 16. Freeze Rule

The following must be fixed before inspecting the final experimental results:

- final financial items
- final language set
- model and model version
- prompt
- decoding parameters
- tokenizer-overlap metric
- primary consistency metric
- primary statistical test

Changes after this point must be documented as exploratory analyses.