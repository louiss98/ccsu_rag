# GPU-Accelerated RAG Pipeline Setup Guide

This guide will help you set up a high-performance RAG (Retrieval-Augmented Generation) pipeline using GPU acceleration with FAISS-GPU and HuggingFace embeddings.

## 🚀 Quick Start

1. **Run the setup script:**
   ```bash
   python setup_gpu_rag.py --validate-all
   ```

2. **Test the installation:**
   ```bash
   python test_faiss_gpu.py
   ```

3. **Run the RAG pipeline:**
   ```bash
   python src/chunker/embedder.py
   ```

## 📋 Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA Compute Capability 6.0+ (GTX 1060 or better)
- At least 4GB GPU memory (8GB+ recommended)
- Sufficient system RAM (16GB+ recommended for large document collections)

### Software Requirements
- Python 3.8-3.11
- NVIDIA GPU drivers (latest recommended)
- CUDA 11.8 or 12.1 (will be installed with PyTorch)

## 🔧 Manual Installation

If the automatic setup script doesn't work, follow these manual steps:

### Step 1: Install NVIDIA Drivers and CUDA

1. **Check GPU:**
   ```bash
   nvidia-smi
   ```
   If this fails, install NVIDIA drivers first.

2. **CUDA is installed automatically with PyTorch** (recommended approach)

### Step 2: Install PyTorch with CUDA

```bash
# For CUDA 12.1 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (if you have compatibility issues)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install FAISS-GPU

```bash
# Method 1: pip (recommended)
pip install faiss-gpu

# Method 2: conda (if pip fails)
conda install -c conda-forge faiss-gpu
```

### Step 4: Install Other Requirements

```bash
pip install -r requirements.txt
```

## 🧪 Testing Your Installation

### Quick Test
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'FAISS GPUs: {faiss.get_num_gpus()}')"
```

### Comprehensive Test
```bash
python test_faiss_gpu.py
```

This will test:
- FAISS installation and GPU availability
- HuggingFace embeddings with GPU support
- Performance comparison between CPU and GPU

### RAG Pipeline Test
```bash
python src/chunker/embedder.py
```

This will:
- Process all documents in `data/unitree_research/`
- Create embeddings using GPU
- Store them in FAISS-GPU vector database
- Perform a test search

## 🔥 Performance Optimization

### GPU Memory Configuration

Edit `src/chunker/gpu_config.py`:

```python
class GPUConfig:
    # Adjust based on your GPU memory
    TORCH_CUDA_MEMORY_FRACTION = 0.8  # Use 80% of GPU memory
    FAISS_GPU_MEMORY_FRACTION = 0.5   # Reserve 50% for FAISS
    BATCH_SIZE = 32  # Increase for more GPU memory
    FAISS_USE_FLOAT16 = True  # Use half precision
```

### Environment Variables

You can also use environment variables:
```bash
export FAISS_GPU_ENABLED=true
export EMBEDDING_DEVICE=cuda
export BATCH_SIZE=64
```

## 📊 Expected Performance Gains

With GPU acceleration, you should see:

- **Embedding Creation**: 5-10x faster than CPU
- **Vector Search**: 10-50x faster than CPU (depending on dataset size)
- **Memory Efficiency**: Using Float16 reduces memory usage by ~50%

## 🛠️ Troubleshooting

### Common Issues

1. **"No GPUs available for FAISS"**
   ```bash
   # Check FAISS-GPU installation
   pip uninstall faiss-cpu faiss-gpu
   pip install faiss-gpu
   ```

2. **"CUDA out of memory"**
   - Reduce `BATCH_SIZE` in `gpu_config.py`
   - Reduce `TORCH_CUDA_MEMORY_FRACTION`
   - Process documents in smaller batches

3. **"Import faiss could not be resolved"**
   ```bash
   # Try conda installation
   conda install -c conda-forge faiss-gpu
   ```

4. **PyTorch not using GPU**
   ```bash
   # Check CUDA version compatibility
   python -c "import torch; print(torch.version.cuda)"
   nvidia-smi
   ```

### Debug Mode

To debug issues, run with verbose output:
```bash
python src/chunker/gpu_config.py  # Shows detailed GPU status
```

## 📁 File Structure

```
ccsu_rag/
├── src/chunker/
│   ├── embedder.py          # Main embedder with GPU support
│   ├── gpu_config.py        # GPU configuration settings
│   └── document_chunker.py  # Semantic chunking
├── data/unitree_research/   # Your documents
├── setup_gpu_rag.py        # Automated setup script
├── test_faiss_gpu.py       # GPU testing script
└── requirements.txt        # All dependencies
```

## 🎯 Usage Examples

### Basic Usage
```python
from src.chunker.embedder import DocumentEmbedder

# Initialize with GPU support
embedder = DocumentEmbedder()

# Process documents
vector_store = embedder.embed_directory("data/go2_robot")

# Save for later use
embedder.save_vector_store("my_vector_db")

# Search
results = embedder.search_similar("humanoid robot control", k=5)
```

### Advanced Configuration
```python
from src.chunker.embedder import DocumentEmbedder
from src.chunker.gpu_config import GPUConfig

# Custom configuration
class MyGPUConfig(GPUConfig):
    BATCH_SIZE = 64
    FAISS_USE_FLOAT16 = True

embedder = DocumentEmbedder(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    use_gpu_config=True
)
```

## 📈 Monitoring Performance

### GPU Usage
```bash
# Monitor GPU usage while running
watch -n 1 nvidia-smi
```

### Memory Usage
```python
from src.chunker.embedder import DocumentEmbedder

embedder = DocumentEmbedder()
memory_info = embedder.get_gpu_memory_info()
print(memory_info)
```

## 🔄 Integration with Existing Code

The GPU-accelerated embedder is designed to be a drop-in replacement:

```python
# Old CPU-only code
# embedder = DocumentEmbedder(embedding_model="...", use_gpu=False)

# New GPU-accelerated code
embedder = DocumentEmbedder()  # GPU enabled by default
```

## 🚀 Next Steps

After setting up GPU acceleration:

1. **Integrate with your LLM pipeline** - Use the search results as context
2. **Scale up** - Process larger document collections
3. **Optimize further** - Fine-tune batch sizes and memory settings
4. **Monitor performance** - Track search latency and throughput

## 📞 Support

If you encounter issues:

1. Run the diagnostic scripts: `python test_faiss_gpu.py`
2. Check GPU compatibility: `python src/chunker/gpu_config.py`
3. Review error messages and logs
4. Consider fallback to CPU if GPU setup is problematic

## 🏆 Performance Benchmarks

Expected performance on common hardware:

| GPU | Embedding Speed | Search Speed | Memory Usage |
|-----|----------------|--------------|--------------|
| RTX 4090 | ~1000 docs/min | <1ms/query | ~6GB |
| RTX 3080 | ~600 docs/min | ~2ms/query | ~8GB |
| RTX 2080 | ~400 docs/min | ~5ms/query | ~6GB |
| GTX 1660 | ~200 docs/min | ~10ms/query | ~4GB |

*Benchmarks based on typical academic papers (10-50 pages each)*
