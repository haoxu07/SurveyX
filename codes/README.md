# Surveyx

Surveyx is an automated tool for generating survey papers based on your local reference library. It leverages large language models (LLMs) to generate outlines, content, and LaTeX source, requiring the user to provide an LLM API URL and key.

## Features
- Generate survey papers from a local reference path
- Modular workflow: preprocessing, outline generation, content creation, post-processing, and LaTeX compilation
- Outputs PDF files

## Prerequisites
- Python 3.10+ (Anaconda recommended)
- All Python dependencies in `requirements.txt`
- LaTeX environment (for PDF compilation):

```bash
sudo apt update && sudo apt install texlive-full
```

## Installation
Install Python dependencies:
```bash
pip install -r requirements.txt
```

## LLM Configuration
All LLM API and key configuration should be set in `src/configs/config.py`. Please edit this file to provide your LLM API URL, token, and model information before running the pipeline.

### Required Configuration in `src/configs/config.py`
Before running Surveyx, you must edit `src/configs/config.py` and fill in the following variables with your own service information:

**For LLM API Service:**
- `REMOTE_URL`: The endpoint URL for your LLM API service
- `TOKEN`: The API token/key for your LLM service

#**For Embedding Service:**
- `DEFAULT_EMBED_ONLINE_MODEL`: The model name or path for your embedding service
- `EMBED_REMOTE_URL`: The endpoint URL for your embedding API service
- `EMBED_TOKEN`: The API token/key for your embedding service

Example:
```python
REMOTE_URL = "https://api.openai.com/v1/chat/completions"
TOKEN = "sk-xxxx..."
DEFAULT_EMBED_ONLINE_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_REMOTE_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_TOKEN = "your embed token here"
```

## Workflow
Each time you run the pipeline, a unique result folder will be created under `outputs/`, named by the task id `outputs/<task_id>` (e.g., `outputs/2025-06-18-0935_contr/`).

You can run the full pipeline with:
```bash
python tasks/offline_run.py --title "Your Survey Title" --key_words "keyword1, keyword2" --ref_path "path/to/your/local/dir"
```

Or execute the workflow step by step (recommended for debugging or advanced usage):
```bash
# Set your task ID
export task_id="your_task_id"

python tasks/workflow/03_gen_outlines.py --task_id $task_id
python tasks/workflow/04_gen_content.py --task_id $task_id
python tasks/workflow/05_post_refine.py --task_id $task_id
python tasks/workflow/06_gen_latex.py --task_id $task_id
```
If any step fails, you can rerun that specific step by providing the same task id.

## Local Reference Format
The local reference documents you provide **must be in Markdown (`.md`) format** and placed together in a single directory. When specifying the `--ref_path` argument, use the path to this directory containing all your `.md` files.

## Output
- All results are saved under `outputs/<task_id>/`
  - `survey.pdf`: Final compiled survey
  - `outlines.json`: Generated outline
  - `latex/`: LaTeX sources
  - `tmp/`: Intermediate files

## Notes
- Log files are in `outputs/logs/` for debugging.
- For advanced usage, run each script in `tasks/workflow/` step by step with the correct task id.

## Open Source Version Notice
This open source version of Surveyx is a simplified edition. It relies entirely on user-provided local reference documents and does not include advanced features such as:
- Keyword expansion and filtering algorithms
- Multimodal image parsing or figure extraction
- Online reference search or automatic data fetching

These advanced modules are only available in the full version of Surveyx, which is hosted by MemTensor (Shanghai) Technology Co., Ltd. If you would like to experience the complete features, please visit our official website: [surveyx.cn](https://surveyx.cn)

---
For questions or issues, please open an issue on the repository.
