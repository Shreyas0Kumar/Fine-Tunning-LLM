# ORACC LLaMA Fine-Tuning Project

This project fine-tunes the Meta-Llama-3.1-8B-Instruct model on the Open Richly Annotated Cuneiform Corpus (ORACC) to create a language model capable of understanding, translating, and generating ancient cuneiform texts.

## Project Overview

The Open Richly Annotated Cuneiform Corpus (ORACC) brings together the work of several Assyriological projects to publish online editions of cuneiform texts. This project leverages this corpus to fine-tune LLaMA 3.1, enabling advanced NLP capabilities for Akkadian, Sumerian, and other ancient Mesopotamian languages.

## Features

- Dataset preparation from VRT (Corpus Workbench Vertical) files
- Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
- Instruction dataset creation for various cuneiform text tasks
- Support for translation, analysis, and text completion

## Directory Structure

```
FINE-TUNNING-LLM/
├── data/                      # Raw ORACC VRT files
│   ├── oracc_cams.vrt
│   ├── oracc_dcclt.vrt
│   ├── oracc_other.vrt
│   ├── oracc_ribo.vrt
│   ├── oracc_rinap.vrt
│   └── oracc_saao.vrt
├── processed_oracc/           # Processed dataset (created during execution)
├── oracc_instructions/        # Instruction dataset (created during execution)
├── fine_tuned_oracc_llama/    # Output directory for the fine-tuned model
├── config.yaml                # Configuration file
├── oracc-dataset-prep.py      # Dataset preparation script
├── oracc-finetuning.py        # Main fine-tuning script
├── peft-finetuning.py         # Alternative PEFT fine-tuning implementation
├── llama-finetuning.py        # Base LLaMA fine-tuning script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU with at least 16GB VRAM
- Access to the Meta-Llama-3.1-8B-Instruct model on Hugging Face
- Hugging Face account with appropriate access token

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/FINE-TUNNING-LLM.git
   cd FINE-TUNNING-LLM
   ```

2. Create a virtual environment:
   ```bash
   python -m venv oracc-env
   source oracc-env/bin/activate  # On Windows: oracc-env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a configuration file:
   ```bash
   cp config.yaml.example config.yaml
   # Edit config.yaml with your Hugging Face token and settings
   ```

## Dataset Preparation

1. Obtain the ORACC VRT files from [Clarino](https://clarino.uib.no/comedi/editor/lb-2018071121)
2. Place the VRT files in the `data` directory
3. Run the dataset preparation script:
   ```bash
   python oracc-dataset-prep.py
   ```

## Fine-Tuning

1. Update the configuration file with your specific settings
2. Run the fine-tuning script:
   ```bash
   python oracc-finetuning.py
   ```

## Usage

After fine-tuning, you can use the model for:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    load_in_8bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./fine_tuned_oracc_llama")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_oracc_llama")

# Example prompt
prompt = "[INST] Translate the following Akkadian text to English: ana bēlīya qibīma umma Sîn-iddinam-ma [/INST]"

# Generate text
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    inputs["input_ids"],
    max_length=200,
    temperature=0.7,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Customization

- Modify `oracc-dataset-prep.py` to change how datasets are prepared
- Adjust LoRA parameters in `config.yaml` for different fine-tuning behavior
- Change training hyperparameters in `config.yaml` to match your GPU resources

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Open Richly Annotated Cuneiform Corpus (ORACC)](http://oracc.museum.upenn.edu/)
- [Meta LLaMA Team](https://ai.meta.com/llama/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT Library](https://github.com/huggingface/peft)

## Citation

If you use this work in your research, please cite:

```
@misc{oracc-llama-finetuning,
  author = {Your Name},
  title = {ORACC LLaMA Fine-Tuning Project},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/FINE-TUNNING-LLM}
}
```