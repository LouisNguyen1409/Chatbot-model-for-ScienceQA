---
base_model: llava-hf/llava-v1.6-mistral-7b-hf
library_name: peft
license: apache-2.0
tags:
- trl
- sft
- generated_from_trainer
model-index:
- name: llava-1.6-7b-hf-final
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llava-1.6-7b-hf-final

This model is a fine-tuned version of [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) on an derek-thomas/ScienceQA.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

## How to use
```python
model_id = "Louisnguyen/llava-1.6-7b-hf-final"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)
model = LlavaNextForConditionalGeneration.from_pretrained(model_id,
                                                      quantization_config=quantization_config,
                                                      torch_dtype=torch.float16)
```
### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1.4e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1
- mixed_precision_training: Native AMP

### Training results

Accuracy ~80%

### Framework versions

- PEFT 0.12.0
- Transformers 4.43.3
- Pytorch 2.2.1+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1
