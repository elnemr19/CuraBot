# CuraBot: A Medical AI Assistant System

CuraBot is an intelligent medical assistant system designed to enhance healthcare support through three integrated modules. The first module is **a Medical Chatbot**, built using large language models (LLMs) fine-tuned on medical data to deliver accurate and conversational responses to a wide range of health-related queries. The second module is **Symptom-Based Image Generation**, which leverages a diffusion model to generate medical illustrations based on user-provided symptom descriptions—offering a visual understanding of potential conditions. The third module focuses on **Brain Tumor Segmentation**, utilizing a fine-tuned YOLOv11 model trained on a custom dataset to detect and segment brain tumors from MRI scans with high precision. Together, these modules form a comprehensive AI-driven system aimed at improving medical awareness, diagnostic support, and user engagement in healthcare.


## 📌 Project Structure

Main Component:

- **Medical Chatbot** — Built by fine-tuning two powerful open-source LLMs:

  - [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) by Alibaba
  
  - [LLaMA-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) by Meta
  
  Both models are fine-tuned on the [ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) dataset, which contains real-world medical Q&A pairs.

## 🧠 Model 1: Qwen2.5-3B-Instruct

### 🔧 Training Pipeline Summary

- **Quantization**: 4-bit via `bitsandbytes` for efficient training

- **PEFT Method**: LoRA (`peft` library)

- **Dataset**: 5,000 samples manually split: 4k for training, 500 for validation, 500 for testing

- **Frameworks**: Transformers, Datasets, PEFT, TRL

### 🛠️ Technologies Used

- `transformers`

- `datasets`

- `bitsandbytes`

- `peft`

- `trl`

- `accelerate`

- `torch`


### You can use my model from this code:

```python
!pip install transformers peft bitsandbytes
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# Load Tokenizer
base_model_name = "Qwen/Qwen2.5-3B-Instruct"
adapter_model_id = "AbdullahAlnemr1/qwen2.5-medical-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


# Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)


# Load LoRA Adapter
model = PeftModel.from_pretrained(base_model, adapter_model_id)
model = model.merge_and_unload()  # merge LoRA weights into base model
model.eval()

# Fix padding issues
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

```python
def generate_response(instruction, input_text=""):
    prompt = f"""<|im_start|>system
You are a highly knowledgeable and accurate medical assistant trained to provide evidence-based medical advice. Answer clearly and concisely using medical best practices. If the question is unclear or potentially harmful to answer, respond with a disclaimer.<|im_end|>
<|im_start|>user
Instruction: {instruction}
{input_text}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return decoded  # Output includes Markdown like **bold**
```

```python
from IPython.display import Markdown

response = generate_response("I'm a 35-year-old woman who has been experiencing persistent abdominal bloating, changes in bowel habits (alternating between constipation and diarrhea), and occasional lower abdominal pain for the past 3 months. I’ve also noticed that I feel more tired than usual and have lost a bit of weight without trying. I don’t have any significant medical history, and these symptoms have gradually worsened over time. Could this be something serious like colon cancer, or is it more likely to be something benign like IBS? What should I do next?")
Markdown(response)  # This will render **bold** text as actual bold
```



## 🧠 Model 2: Meta LLaMA 3.1-8B

### 🔧 Training Summary

- Similar fine-tuning procedure to Qwen, using the same dataset.

- LoRA used for parameter-efficient training.

- Currently in progress (additional README updates will follow).

## 📊 Dataset

- **Source**: [lavita/ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)

- **Description**: Contains over 100,000 real user medical questions and doctor responses

- **Fields**: instruction, input, output



