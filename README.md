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

## Chat Template
```python
CHAT_TEMPLATE='''<<SYS>>
A chat between an user and an artificial intelligence assistant about Science Question Answering. The assistant gives helpful, detailed, and polite answers to the user's questions.
Based on the image, question and hint, please choose one of the given choices that answer the question.
Give yourself room to think by extracting the image, question and hint before choosing the choice.
Don't return the thinking, only return the highest accuracy choice.
Make sure your answers are as correct as possible.
<</SYS>> 
{% for tag, content in messages.items() %}
{% if tag == 'real_question' %}
Now use the following image and question to choose the choice:
{% for message in content %}
{% if message['role'] == 'user' %}[INST] USER: {% else %}ASSISTANT: {% endif %}
{% for item in message['content'] %}
{% if item['type'] == 'text_question' %}
Question: {{ item['question'] }}
{% elif item['type'] == 'text_hint' %}
Hint: {{ item['hint'] }}
{% elif item['type'] == 'text_choice' %}
Choices: {{ item['choice'] }} [/INST]
{% elif item['type'] == 'text_solution' %}
Solution: {{ item['solution'] }}
{% elif item['type'] == 'text_answer' %}
Answer: {{ item['answer'] }}{% elif item['type'] == 'image' %}<image>
{% endif %}
{% endfor %}
{% if message['role'] == 'user' %}
{% else %}
{{eos_token}}
{% endif %}{% endfor %}{% endif %}
{% endfor %}'''
```

## How to use
```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests


model_id = "Louisnguyen/llava-1.6-7b-hf-final"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)
model = LlavaNextForConditionalGeneration.from_pretrained(model_id,
                                                      quantization_config=quantization_config,
                                                      torch_dtype=torch.float16)
model.to("cuda:0")
processor = LlavaNextProcessor.from_pretrained(model_id)

image = example["image"]
question = example["question"]
choices = example["choices"]
hint = example["hint"]

messages_answer = {
        "real_question": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text_question", "question": question},
                    {"type": "text_hint", "hint": hint},
                    {"type": "text_choice", "choice": ' or '.join(choices)},
                ]
            }
        ]
    }
# Apply the chat template to format the messages for answer generation
text_answer = processor.tokenizer.apply_chat_template(messages_answer, tokenize=False, add_generation_prompt=True)
# Prepare the inputs for the model to generate the answer
inputs_answer = processor(text=[text_answer.strip()], images=image, return_tensors="pt", padding=True).to('cuda')
# Generate text using the model for the answer
generated_ids_answer = model.generate(**inputs_answer, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
# Decode the generated text for the answer
generated_texts_answer = processor.batch_decode(generated_ids_answer[:, inputs_answer["input_ids"].size(1):], skip_special_tokens=True)
print(generated_texts_answer)
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
