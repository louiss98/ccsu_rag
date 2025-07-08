This is a prototype Retrieval-Augmented Generation (RAG) application developed for Central Connecticut State University IT. It is intended for internal use only. All responses are AI-generated and do not represent the views of any group, department, individual, or organization.

## Hardware Requirements

- **GPU:** Minimum 24GB VRAM (NVIDIA RTX 3090, A5000, or better recommended)
- **CPU:** 8-core modern processor recommended
- **RAM:** At least 32GB system memory
- **Storage:** SSD strongly recommended

## Installation

First, install the required dependencies listed in `requirements.txt`. The following example uses Anaconda to set up the Python environment:

```bash
conda create -n ccsu_rag_env python=3.10
conda activate ccsu_rag_env
pip install -r requirements.txt
```

Next, sign in to your HuggingFace account with access to the required language models. You may need to request access on the HuggingFace model pages. The models used are open source but gated to prevent unauthorized access.

**Models:**
- [Meta-Llama 3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [Llama Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

## Data Sources

A sample data source is included in the `data` folder. To add a new data source, place your files in a new folder inside the `data` directory.

**Supported file types:** `.docx`, `.pdf`, `.txt`

To process your data, run the chunker script:

```bash
python src/chunker/embedder.py --input_folder "data/new_data_source" --output_store "new_data_source_vector_store"
```

**Note:** FAISS-CPU is used for chunking. Please ensure your processor is capable.

**Application entry point:** `tui1.py`