# Code Completion for NLP++ with CodeLlama
---

This repository contains code for creating a dataset of [NLP++](https://visualtext.org/) code examples (final dataset can be found [here](https://huggingface.co/datasets/AshtonIsNotHere/nlp_pp_code_dataset) as well as code to fine-tune CodeLlama (7B- and 13B-parameter models) for autocompletion. For more information on CodeLlama see the [paper](https://arxiv.org/abs/2308.12950) and check out the models on [HF](meta-llama/Llama-2-7b-hf). 

## Overview

Due to the relative paucity of NLP++ code (compared to, for example, Python or Java), compiling a dataset of sufficient size ([~500B tokens](https://arxiv.org/pdf/2308.12950.pdf#subsection.2.2)) to pretrain an LLM is likely not possible (without even considering computational requirements). Thus the most promising option to create a code completion model for NLP++ is to adapt an existing model. To this end there are a few challenges to address: 

### Choosing a foundation model
The recently released CodeLlama is chosen as a foundation model due to its reported state-of-the-art performance and availability/accessibility. The 7B- and 13B-parameter base versions (i.e. not adapted to Python code or instruction tasks) are chosen in particular due to their lower computational cost compared to the larger versions (34B+). 

### Overcoming the gap between the source and target domains
Due to the uniqueness of NLP++ code and its (assumed) absence in the training dataset, there is likely a significant gap between the source and target distributions. To attempt to overcome this gap, the foundation model is finetuned on a Causal LM objective using a dataset of 1500+ NLP++ code examples compiled from GitHub and the help pages of the Visual Text website. Whether this is the optimal method to leverage the existing data and whether the distribution gap is surmountable at all with this dataset remain open questions.

### Training Large Language Models
Despite being the smallest of the CodeLlama releases, the 7B- and 13B-parameter models are still too large to train on a single 40gb A100 (with bf16, an input size of 1024, b.s. of 1, Adam optimizer, and gradient checkpointing) without some computational optimizations. Instead of implementing a more efficient adaptation method (for a good overview of these, see [here](https://github.com/huggingface/peft)), both models are fine-tuned in a multinode multi-gpu setup with Zero Redundancy Optimizer (you can find the implementation and papers [here](https://github.com/microsoft/DeepSpeed)) with optimizer and parameter offloading.

The final model weights can be found [here](https://huggingface.co/AshtonIsNotHere/CodeLlama_7B_nlp_pp). For some examples on implementing these for code-completion take a look at [HF Code Autocompletion extension](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode) and [_Continue_](https://continue.dev/docs/walkthroughs/codellama).

