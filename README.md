# Library AI Chatbot

**Author:** Adriel Agustinus Jie  
**Assessment:** Option 1 — RAG Implementation  
**Repository:** [adrieljie/RAG-Implementation](https://github.com/adrieljie/RAG-Implementation)

## Problem Statement
Readers may want to understand more about the characters, events, and motivations in a book, but they may not always have someone available to discuss it with. Finding passages that answer their questions can also require searching through different parts of the book.

This project builds a RAG-based book assistant that allows readers to ask questions about a collection of books. It retrieves relevant passages and generates answers with source references so readers can check the supporting evidence.

## Project Files
```RAG-Implementation/
├── README.md
├── requirements.txt
├── RAG_baseline.ipynb
├── RAG_experiment.ipynb
├── review.ipynb
└── docs/
    ├── alice_in_wonderland.txt
    ├── frankenstein.txt
    ├── pride_and_prejudice.txt
    └── sherlock_holmes.txt
```

- [RAG_baseline.ipynb](RAG_baseline.ipynb): baseline implementation and saved evaluation outputs.
- [RAG_experiment.ipynb](RAG_experiment.ipynb): experiment implementation, where last saved code is similarity score threshold with `0.7` threshold score.
- [review.ipynb](review.ipynb): design decisions, limitations, future works, saved examples, evaluation and manual reviews, and experiment results.
- [requirements.txt](requirements.txt): direct dependencies with pinned versions.

The local environment, downloaded models, and generated vector database are not required in the repository. The database is created through the Load Data menu.

## Dataset & Preprocessing
The dataset contains five English-language books from Project Gutenberg, stored as UTF-8 `.txt` files.

| Book | Author | Filename | File size |
|---|---|---|---:|
| [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11) | Lewis Carroll | `alice_in_wonderland.txt` | 174 KB |
| [Frankenstein](https://www.gutenberg.org/ebooks/84) | Mary Wollstonecraft Shelley | `frankenstein.txt` | 448 KB|
| [Pride and Prejudice](https://www.gutenberg.org/ebooks/1342) | Jane Austen | `pride_and_prejudice.txt` | 772 KB |
| [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661) | Arthur Conan Doyle | `sherlock_holmes.txt` | 607 KB |
| [The Great Gatsby](https://www.gutenberg.org/ebooks/64317) | Francis Scott Fitzgerald | `the_great_gatsby.txt` | 307 KB |

Total dataset size: 5 files, approximately 2.3 MB.

The books contain different characters, events, and passages, making them suitable for testing retrieval across a collection. Plain text was selected to avoid PDF layout and OCR problems.

The original files retain Project Gutenberg's usage notices and license. Cleaning removes boilerplate from the searchable text, not from the original files.

## Architecture

```text
INGESTION
TXT files → duplicate check → text loading and metadata extraction → preprocessing → chunks → BGE embeddings → Chroma

QUESTION ANSWERING
Question → BGE query embedding → top-5 similarity retrieval → full passages + question + prompt rules → Qwen → answer + retrieved source references
```

The embedding model represents questions and passages as vectors for retrieval. Qwen receives the retrieved text and generates the answer, it does not search Chroma directly.

Ingestion, Preprocessing, and Metadata

`load_document()` reads `.txt` files directly inside `DOCS_PATH`. It does not recursively load subfolders. `TextLoader` uses UTF-8 decoding.

Before preprocessing, the program extracts `title`, `author`, `release_date`, and `language` from the Gutenberg headers. Missing values become `"Unknown"`. It also preserves `source` and adds `hash_file`, a SHA-256 hash of the original file bytes.

The preprocessing function:

1. Keeps the content between the Gutenberg START and END markers when both match.
2. Removes lines containing only asterisks.
3. Removes standalone `[Illustration]` markers.
4. Reduces repeated blank lines and trims leading and trailing whitespace.

It does not remove every illustration caption, multiline illustration block, or table of contents.

`RecursiveCharacterTextSplitter` uses a maximum size of 1,000 characters, 0 overlap, and separators in this order: `"\n\n"`, `"\n"`, `" "`, `""`. These are character counts, not token counts.

Chunks inherit document metadata and receive a numeric `chunk_id`, starting at 0 within each source file. A filename and chunk ID together identify a passage. Plain-text files do not provide reliable page numbers.

Duplicate prevention checks the file hash against stored metadata and against hashes already seen in the current loading operation. Identical file contents are skipped even if the filename differs.

## Retrieval, Generation, and Source References

The baseline uses similarity search with cosine distance, returning up to five chunks. No similarity threshold is applied in the baseline.

The complete retrieved chunks are joined and supplied with the question. The system prompt asks Qwen to use only the context, combine relevant passages, consider the order of events, remain concise but complete, and avoid inventing information. The user prompt also asks it to say it does not know when the documents do not provide an answer.

With the default settings, the retrieved context contains up to roughly 5,000 characters, plus separators. The question and instructions add more input. There is no separate token-based context trimming.

Generation uses `max_new_tokens=512`, `do_sample=False`, and `return_full_text=False`. The 512-token limit applies to the generated output, not the input.

If retrieval returns no passages, the function returns `"I couldn't find relevant information."` without calling the LLM. When passages are returned but are irrelevant, abstention depends on the model following the prompt.

The output prints the title, author, chunk ID, and a 250-character evidence preview for each retrieved passage. The LLM receives the full passage. These references describe the retrieved context, not verified citations attached to individual answer claims.

Run this example in RAG_baseline.ipynb after initialization and document ingestion to inspect the full retrieved passages.

```python
answer, retrieved_docs = ask_question(
    "Who does Sherlock Holmes introduce as his friend and associate to the veiled lady?"
)

for doc in retrieved_docs:
    print(doc.metadata["source"], doc.metadata["chunk_id"])
    print(doc.page_content)
```

## Tools & Models
Component | Model or tool | Reason for selection |
|---|---|---|
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` | An instruction-tuned model suited to answering questions from a supplied context. Its approximately 1.5B parameters make it a more manageable starting point for a laptop than a 7B model at the same numerical precision. This does not imply better answer quality than larger models. |
| Embedding model | `BAAI/bge-base-en-v1.5` | Supports English passage retrieval, matching the collection and task. Its published retrieval benchmark score is higher than the small version, while its embeddings have fewer dimensions than the large version. It was selected as a balance between benchmark performance and vector storage requirements. |
| Vector store | ChromaDB | Stores embeddings, text, and metadata locally, supports metadata filtering for duplicate checks, and preserves the collection between sessions without a separate database server. |
| Framework | LangChain integrations | Connect loading, chunking, embedding, retrieval, and generation with compatible components, reducing the code needed to connect them manually. The trade-off is additional dependencies and version compatibility requirements. |
| Text splitter | `RecursiveCharacterTextSplitter` | Tries natural text boundaries before smaller splits and does not require consistent chapter headings. Unlike cutting at fixed character positions, it tries to preserve paragraph and word boundaries, although complete ideas can still cross chunks. |
| Retrieval method | Similarity search with cosine distance, `k=5` | Provides a simple, inspectable baseline for semantic retrieval. It limits context to five passages without first tuning a score cutoff. It can still return irrelevant passages when the collection has no answer. |

The choices prioritize a manageable local baseline. They have not all been compared experimentally with alternatives, so they are not presented as the best-performing options.

Technical references: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [BGE model card and benchmarks](https://huggingface.co/BAAI/bge-base-en-v1.5), [Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma), and [recursive splitter documentation](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).

## Configuration
```python DOCS_PATH = "./docs/"
DB_PATH = "./db/ChromaDB/"
COLLECTION_NAME = "books_dedup"

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEVICE = 0

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
K = 5
MAX_NEW_TOKENS = 512
```
Setting | Purpose |
|---|---|
| `DOCS_PATH` | Folder containing the `.txt` books. |
| `DB_PATH` / `COLLECTION_NAME` | Location and collection used by Chroma. |
| `EMBEDDING_MODEL` | Model used to embed documents and questions. |
| `LLM_MODEL` | Model used to generate answers. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Intended chunking settings; see the implementation note below. |
| `DEVICE` | Device used to run the LLM. |
| `MAX_NEW_TOKENS` | Maximum generated answer length in tokens. |
| `K` | Maximum number of retrieved passages. |

In the development environment, `DEVICE=0` resolved to `mps:0`, using the Apple GPU. Device selection can differ between systems. Use `"mps"` explicitly for an available Apple MPS device, or `-1` for CPU execution. Check the loaded LLM device with:

```python
print(pipeline.pipeline.device)
```

This device setting applies to the LLM, it is not explicitly passed to the embedding model.

The provider is local Hugging Face model execution. No paid API, API key, or `.env` file is required. The experiment notebook has separate settings, baseline configuration changes do not automatically update it.

After changing configuration, rerun the affected cells. Changing the embedding model requires rebuilding the embeddings in a fresh collection. Changing preprocessing or chunking also requires rebuilding the index and checking the reference chunk IDs used for evaluation.

## Setup and Run Commands

The inspected environment uses Python 3.10.0 on an Apple Silicon Mac. LLM execution was checked on MPS (`mps:0`).

Allow several GB of disk space and free memory for model downloads and execution. An exact minimum RAM requirement has not been measured. CPU execution is configurable but may be slower.

Internet access is needed for dependency installation and initial model downloads. Downloaded models are cached locally.

**1. Clone the repository and create an environment** (macOS/Linux):

```bash
git clone https://github.com/adrieljie/RAG-Implementation.git
cd RAG-Implementation
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies:**

```bash
python -m pip install -r requirements.txt
```

The dependency file pins versions, including `torch==2.14.1`, `transformers==5.20.1`, `sentence-transformers==6.1.0`, `chromadb==1.5.9`, and `ragas==0.3.9`. The complete list is in [requirements.txt](requirements.txt). The `langchain-community==0.4.1` pin retains a module needed by the installed Ragas version.

**3. Run the CLI**

From the repository directory, run:

`python RAG.py`

The application displays a text menu:

```Library AI Chatbot!

Menu:
1. Load Data
2. Ask LLM
3. Exit
```

1. Select 1. Load Data to process the books and store their embeddings in Chroma.

2. Select 2. Ask LLM to enter questions about the books.

3. Type `quit` or `Quit` to return to the main menu.

4. Select 3. Exit to close the application.

On the first run, model loading may take longer because the models need to be downloaded. The database is stored locally and can be reused on later runs.

`RAG.py` runs the application only. It does not execute retrieval or generation evaluation.

**4. Run Evaluation Separately**

Open `RAG_baseline.ipynb` in VS Code using the Python and Jupyter extensions, then select the .venv interpreter as the notebook kernel.

Alternatively, launch JupyterLab:

```bash
python -m pip install jupyterlab
python -m jupyter lab RAG_baseline.ipynb
```

Run the notebook setup and function definitions first. Make sure the books have been ingested into the database selected by the notebook. Then run Retrieval Evaluation followed by Generation Evaluation, including their data and helper cells.

The notebook must initialize its own models and database connection, variables from the CLI process are not shared with it.

Generation evaluation uses the local Qwen model as the evaluator and BGE embeddings, with one worker and no automatic retries. It can take longer than answering an individual question.

Saved outputs are available in the notebooks so reviewers can inspect the results without rerunning evaluation.