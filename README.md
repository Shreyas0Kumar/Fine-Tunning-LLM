# Guide to Fine-Tuning LLaMA 3.1 on ORACC Cuneiform Corpus

This guide will walk you through the process of fine-tuning the Meta-Llama-3.1-8B-Instruct model on the Open Richly Annotated Cuneiform Corpus (ORACC) to create a model that understands and can work with Akkadian, Sumerian, and other ancient languages.

## 1. Environment Setup

First, set up a Python environment with the necessary dependencies:

```bash
# Create a virtual environment
python -m venv oracc-env
source oracc-env/bin/activate  # On Windows: oracc-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `requirements.txt` file with:

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0
bitsandbytes>=0.39.0
huggingface-hub>=0.15.0
tensorboard>=2.12.0
peft>=0.4.0
sentencepiece>=0.1.99
pyyaml>=6.0
tqdm>=4.65.0
pandas>=1.5.0
```

## 2. Download and Prepare the ORACC Dataset

The ORACC dataset is available from the Clarino repository. You need to download the VRT files:

```bash
# Create directories
mkdir -p oracc_files
mkdir -p processed_oracc
mkdir -p oracc_instructions

# Download the VRT files
# You can either download them manually from the website or use wget/curl
# For example:
# wget -P oracc_files https://clarino.uib.no/comedi/download/oracc_cams.vrt
# wget -P oracc_files https://clarino.uib.no/comedi/download/oracc_dcclt.vrt
# wget -P oracc_files https://clarino.uib.no/comedi/download/oracc_other.vrt
# wget -P oracc_files https://clarino.uib.no/comedi/download/oracc_ribo.vrt
# wget -P oracc_files https://clarino.uib.no/comedi/download/oracc_rinap.vrt
# wget -P oracc_files https://clarino.uib.no/comedi/download/oracc_saao.vrt
```

## 3. Process the Dataset

Run the dataset preparation script to convert the VRT files into a format suitable for fine-tuning:

```bash
python oracc-dataset-prep.py
```

This script:
1. Parses the VRT files to extract text and metadata
2. Creates a regular text dataset
3. Creates an instruction dataset with various prompts for fine-tuning

## 4. Update Configuration

Before running the fine-tuning:

1. Open the `oracc_config.yaml` file
2. Replace `your_huggingface_token` with your actual Hugging Face token
3. Adjust training parameters based on your hardware capabilities

## 5. Run the Fine-Tuning

```bash
python oracc-finetuning.py
```

This script uses LoRA (Low-Rank Adaptation) for efficient fine-tuning. The process might take several hours to complete, depending on your hardware.

## 6. Hardware Requirements

For fine-tuning with LoRA on LLaMA 3.1 8B:
- GPU with at least 16GB VRAM (e.g., NVIDIA A100, V100, or RTX 4090)
- 32GB+ system RAM
- About 50GB of free disk space

## 7. Using the Fine-Tuned Model

After training, you can use the model to:
- Translate ancient texts
- Analyze cuneiform inscriptions
- Generate text in Akkadian/Sumerian style
- Answer questions about ancient Mesopotamian cultures

Example usage:

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

## 8. Customization Options

You can customize the fine-tuning process by:

1. **Modifying the instruction templates**:
   - Edit the `create_instruction_dataset` function in `oracc-dataset-prep.py`
   - Add more domain-specific prompts

2. **Adjusting LoRA parameters**:
   - Change the rank (`r`) and alpha parameters
   - Target different modules in the model

3. **Training parameters**:
   - Increase/decrease epochs
   - Adjust learning rate
   - Change batch size and gradient accumulation steps

## 9. Understanding the VRT Format

VRT (Corpus Workbench Vertical) files contain token-per-line information with annotations:

```
<text id="oracc.cams.P295869" language="Akkadian" period="Old Babylonian">
<s>
awīlu    N    man
kīma    CONJ    like
</s>
</text>
```

The dataset preparation script extracts:
- Text content
- Metadata (language, period, etc.)
- Annotations when available

## 10. Troubleshooting

If you encounter issues:

1. **Out of Memory errors**:
   - Reduce batch size
   - Increase gradient accumulation steps
   - Use `load_in_4bit=True` instead of `load_in_8bit=True`
   - Reduce sequence length

2. **CUDA issues**:
   - Ensure you have the correct CUDA toolkit installed
   - Update GPU drivers

3. **Slow training**:
   - Enable mixed precision training
   - Use a smaller subset of the data for testing

4. **Poor results**:
   - Increase the number of epochs
   - Adjust learning rate
   - Use more instruction examples

## 11. Additional Resources

- [ORACC Website](http://oracc.museum.upenn.edu/) - The official ORACC project site
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index) - For Parameter-Efficient Fine-Tuning details
- [Transformers Documentation](https://huggingface.co/docs/transformers/index) - For more options and customizations
