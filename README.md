# Library AI Chatbot

**Author:** Adriel Agustinus Jie  
**Assessment:** Option 1 — RAG Implementation  
**Repository:** [adrieljie/RAG-Implementation](https://github.com/adrieljie/RAG-Implementation)

## Completed Work
### Problem Statement
Readers may want to understand more about the characters, events, and motivations in a book, but they may not always have someone available to discuss it with. Finding passages that answer their questions can also require searching through different parts of the book.

This project builds a RAG-based book assistant that allows readers to ask questions about a collection of books. It retrieves relevant passages and generates answers with source references so readers can check the supporting evidence.

### Project Files
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
- [review.ipynb](review.ipynb): saved examples, manual reviews, and experiment notes.
- [requirements.txt](requirements.txt): direct dependencies with pinned versions.

The local environment, downloaded models, and generated vector database are not required in the repository. The database is created through the Load Data menu.

### Dataset & Preprocessing
The dataset contains four English-language books from Project Gutenberg, stored as UTF-8 `.txt` files.

| Book | Author | Filename | File size |
|---|---|---|---:|
| [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11) | Lewis Carroll | `alice_in_wonderland.txt` | 174,311 bytes |
| [Frankenstein](https://www.gutenberg.org/ebooks/84) | Mary Wollstonecraft Shelley | `frankenstein.txt` | 448,885 bytes |
| [Pride and Prejudice](https://www.gutenberg.org/ebooks/1342) | Jane Austen | `pride_and_prejudice.txt` | 772,386 bytes |
| [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661) | Arthur Conan Doyle | `sherlock_holmes.txt` | 607,606 bytes |

Total dataset size: 4 files, 2,003,188 bytes, approximately 2 MB.

The books contain different characters, events, and passages, making them suitable for testing retrieval across a collection. Plain text was selected to avoid PDF layout and OCR problems.

The original files retain Project Gutenberg's usage notices and license. Their notices explain the applicable reuse terms and that users outside the United States need to check local copyright rules. Cleaning removes boilerplate from the searchable text, not from the original files.

### Architecture

```text
INGESTION
TXT files → duplicate check → text loading and metadata extraction
          → preprocessing → chunks → BGE embeddings → Chroma

QUESTION ANSWERING
Question → BGE query embedding → top-5 similarity retrieval
         → full passages + question + prompt rules
         → Qwen → answer + retrieved source references
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

`RecursiveCharacterTextSplitter` uses a maximum size of **1,000 characters**, **0 overlap**, and separators in this order: `"\n\n"`, `"\n"`, `" "`, `""`. These are character counts, not token counts.

Chunks inherit document metadata and receive a numeric `chunk_id`, starting at **0 within each source file**. A filename and chunk ID together identify a passage. Plain-text files do not provide reliable page numbers.

Duplicate prevention checks the file hash against stored metadata and against hashes already seen in the current loading operation. Identical file contents are skipped even if the filename differs.

### Retrieval, Generation, and Source References

The baseline uses similarity search with cosine distance, returning up to five chunks. No similarity threshold is applied in the baseline.

The complete retrieved chunks are joined and supplied with the question. The system prompt asks Qwen to use only the context, combine relevant passages, consider the order of events, remain concise but complete, and avoid inventing information. The user prompt also asks it to say it does not know when the documents do not provide an answer.

With the default settings, the retrieved context contains up to roughly 5,000 characters, plus separators. The question and instructions add more input. There is no separate token-based context trimming.

Generation uses `max_new_tokens=512`, `do_sample=False`, and `return_full_text=False`. The 512-token limit applies to the generated output, not the input.

If retrieval returns no passages, the function returns `"I couldn't find relevant information."` without calling the LLM. When passages are returned but are irrelevant, abstention depends on the model following the prompt.

The output prints the title, author, chunk ID, and a 250-character evidence preview for each retrieved passage. The LLM receives the full passage. These references describe the retrieved context, not verified citations attached to individual answer claims.

To inspect full evidence after initialization and ingestion:

```python
answer, retrieved_docs = ask_question(
    "Who does Sherlock Holmes introduce as his friend and associate to the veiled lady?"
)

for doc in retrieved_docs:
    print(doc.metadata["source"], doc.metadata["chunk_id"])
    print(doc.page_content)
```

### Tools & Models
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

### Configuration
``` DOCS_PATH = "./docs/"
DB_PATH = "./db/ChromaDB/"
COLLECTION_NAME = "books_dedup"

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEVICE = ""

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

This device setting applies to the LLM; it is not explicitly passed to the embedding model.

The provider is local Hugging Face model execution. No paid API, API key, or `.env` file is required. The experiment notebook has separate settings; baseline configuration changes do not automatically update it.

After changing configuration, rerun the affected cells. Changing the embedding model requires rebuilding the embeddings in a fresh collection. Changing preprocessing or chunking also requires rebuilding the index and checking the reference chunk IDs used for evaluation.

### Setup and Run Commands

The inspected environment uses Python 3.10.0 on an Apple Silicon Mac. LLM execution was checked on MPS (`mps:0`). This setup does not require an NVIDIA GPU.

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

**3. Open the notebook:**

In VS Code, use the Python and Jupyter extensions, open `RAG_baseline.ipynb`, and select `.venv` as the kernel. If using the existing development environment, select `.venv-arm`.

Alternatively:

```bash
python -m pip install jupyterlab
python -m jupyter lab RAG_baseline.ipynb
```

**4. Run the baseline:**

1. Work from the repository directory so relative paths resolve correctly.
2. Run the imports, configuration, database initialization, and function definitions under Load Documents.
3. Run the model-loading cell and `ask_question()` definition under Retrieval.
4. Skip the evaluation sections for a normal chat session.
5. Run the function definition and final call under Main.
6. Select 1. Load Data to build the database, then 2. Ask LLM.
7. Type `quit` to return to the menu, or select 3. Exit.

```text
Library AI Chatbot!

Menu:
1. Load Data
2. Ask LLM
3. Exit
```

The text menu runs inside the notebook, there is no separate terminal CLI script. Avoid Run All for a normal session because it also executes evaluation.

To evaluate the baseline, initialize the system and load the books first, exit the menu, and then run the Retrieval Evaluation and Generation Evaluation sections in order. Ragas uses the local Qwen model, the BGE embedding wrapper, one worker, and no automatic retries. A wrapper runs synchronous model generation in a thread for Ragas's async interface.

### AI Assistance and References

I built and integrated the baseline pipeline, including document loading, text chunking, embedding generation, retrieval, and answer generation. I used AI assistance to help implement preprocessing and evaluation metrics, prepare documentation, and review and debug parts of the baseline code. I ran the system and manually checked retrieved passages and generated answers against the source books.

Model specifications and tool explanations are based on the following documentation. The architecture diagram above describes this project's implementation.

- [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [BGE-base-en-v1.5 model card and benchmarks](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [Recursive character splitting](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)
- [Ragas documentation](https://docs.ragas.io/)

## 2. Observed Results

The results below come from saved notebook outputs. They are not new measurements made while writing this README, and the latest configuration refactor has not been independently rerun as part of this documentation update.

### Baseline Evaluation

Five selected factual questions were evaluated: four about Alice and one about Sherlock Holmes. Each has a selected reference chunk. Generation evaluation uses the first reference answer listed for each question.

| Question | First reference rank | Hit@5 | Reciprocal rank |
|---|---:|---:|---:|
| Why does Alice follow the White Rabbit? | 3 | 1 | 0.333 |
| What opens the small door behind the curtain? | 1 | 1 | 1.000 |
| Why can't Alice climb the table leg to reach the key? | 1 | 1 | 1.000 |
| Who wakes Alice at the end of her adventure? | Not found | 0 | 0.000 |
| Who does Sherlock Holmes introduce as his friend and associate to the veiled lady? | 1 | 1 | 1.000 |

| Metric | Saved result | Interpretation |
|---|---:|---|
| Hit@5 | 80.00% | A reference chunk was retrieved for four of five questions. |
| MRR@5 | 0.667 | Mean reciprocal rank of the first reference chunk; a miss contributes zero. |
| Mean faithfulness | 0.667 | Automated assessment of support for answer claims in retrieved context. |
| Mean answer correctness | 0.656 | Automated assessment against reference answers, using factual comparison and semantic similarity. |

Hit@5 and MRR@5 provide a simple initial check of finding and ranking reference evidence. They do not measure the relevance of every retrieved chunk or the completeness of evidence for multi-passage questions.

Generation evaluation uses the same local Qwen model as generation, so it is self-evaluation. These scores are not percentages of correctly answered questions.

### Saved Examples and Failure Analysis

Answers below reproduce saved model outputs. Evidence excerpts are shortened from the identified passages, with line breaks normalized for readability. Full passages can be inspected using `retrieved_docs`. A reference passage that was missed is explicitly distinguished from retrieved evidence.

#### Example 1 — Supported answer

**Question:** Who does Sherlock Holmes introduce as his friend and associate to the veiled lady?

**Generated answer:** Sherlock Holmes introduces Dr. Watson as his friend and associate to the veiled lady.

**Retrieved passages, in order:** `sherlock_holmes.txt` — chunks 425, 451, 587, 406, 368.

**Evidence from chunk 425:**

> This is my intimate friend and associate, Dr. Watson, before whom you can speak as freely as before myself.

**Review:** The answer identifies the correct person, supported directly by the first retrieved passage.

#### Example 2 — Relevant evidence retrieved, answer incomplete

**Question:** What opens the small door behind the curtain?

**Generated answer:** Alice opens the small door behind the curtain.

**Retrieved passages, in order:** `alice_in_wonderland.txt` — chunk 8; `sherlock_holmes.txt` — chunks 723, 52, 645; `alice_in_wonderland.txt` — chunk 9.

**Evidence from Alice, chunk 8:**

> she tried the little golden key in the lock, and to her great delight it fitted!

**Review:** The answer should identify the little golden key. The correct evidence was ranked first, but the answer names Alice instead of the object used. This is a generation failure despite a retrieval hit.

The automated answer-correctness score was approximately 0.900, illustrating why evaluator scores require manual checking.

#### Example 3 — Missing reference evidence and incorrect generation

**Question:** Who wakes Alice at the end of her adventure?

**Generated answer:** Alice doesn't wake anyone at the end of her adventure. The text describes Alice's journey through Wonderland, including meeting various characters such as the Mad Hatter, the March Hare, and the Dormouse. However, there is no mention of anyone waking Alice at the end of her adventure.

**Retrieved passages, in order:** `alice_in_wonderland.txt` — chunks 7, 51, 40, 96, 52.

**Excerpt from retrieved chunk 7:**

> Alice was not a bit hurt, and she jumped up on to her feet in a moment

**Review:** This retrieved passage concerns an earlier event. The answer also reverses the question from who wakes Alice to whether Alice wakes someone. The reference answer is her sister, supported by chunk 167, which was not retrieved*

> “Wake up, Alice dear!” said her sister

This example contains both a retrieval miss and a generation error. Chunk 167 is provided here for evaluation, not presented as evidence the model received.

### Optional Exploration: Similarity Threshold

**What was tried:** Adding `similarity_score_threshold`, with thresholds of `0.7`, `0.5`, and `0.3`.

**Why:** The baseline can return passages even for questions unrelated to the collection.

**How:** The threshold notebook keeps `k=5` and applies the selected threshold in both question answering and retrieval evaluation. Recorded comparisons are in `review.ipynb`; the saved experiment code currently uses `0.7` and a separate database folder.

| Configuration | Hit@5 | MRR@5 | Mean faithfulness | Mean answer correctness |
|---|---:|---:|---:|---:|
| Baseline, no threshold | 80.00% | 0.667 | 0.667 | 0.656 |
| Threshold 0.7 | 20.00% | 0.067 | 0.200 | 0.193 |
| Threshold 0.5 | 80.00% | 0.667 | 0.667 | 0.656 |
| Threshold 0.3 | 80.00% | 0.667 | 0.667 | 0.656 |

In the recorded five-question evaluation, `0.7` removed enough evidence to reduce retrieval results. Thresholds `0.5` and `0.3` did not improve the recorded aggregate metrics over the baseline.

A separate saved example at threshold `0.5` asked “What is the weather in Jakarta?” It retrieved **zero passages** and returned “I couldn't find relevant information.” No source should be attributed to this fallback because no passage was retrieved.

These observations support threshold filtering as a possible way to reject some unrelated results. They do not establish a universally suitable threshold or an improvement in answer quality.

An earlier narrative in `review.ipynb` says that `0.7` returned no documents for all example inputs. That statement must not be generalized to the five-question evaluation, which records one hit. The summary here follows the numerical outputs and distinguishes the test sets.

### Optional Exploration: Follow-up Question Rewriting

The recorded experiment used chat history to rewrite references such as “she,” “it,” and “there” into standalone questions before retrieval.

The notes report some successful rewrites, but also unnecessary connections to earlier questions. A saved failure rewrote “Why does she follow him?” into “She follows the White Rabbit because he summons her into his lair in Wonderland.” This introduces an answer and unsupported information instead of preserving the question.

The experiment did not consistently preserve user intent. It is documented as exploration and is not enabled in the baseline. It has not received the same quantitative evaluation as the retrieval baseline.

## 3. Limitations and Uncertainties

### Known Limitations

- **Retrieval:** Similarity can match the right character but the wrong event. Top-k retrieval can return irrelevant passages.
- **Generation:** Qwen can omit the requested fact or make unsupported claims even when relevant evidence is available.
- **Attribution:** Displayed references list retrieved passages, they do not prove that each answer claim is supported.
- **Chunking and cleaning:** Zero overlap can separate related information. Contents pages and some illustration text remain.
- **Evaluation:** Five selected questions cover only Alice and Sherlock Holmes, not the complete collection. The same small model generates and evaluates answers.
- **Document updates:** Modified files receive new hashes, old versions are not automatically removed.
- **Reproducibility:** Chunk IDs depend on preprocessing and source versions. Dependencies are pinned directly, but model snapshot revisions and all transitive dependency versions are not locked.
- **Portability:** Saved outputs demonstrate execution in the development environment, a fresh installation on every supported device has not been verified. The interface is a notebook menu rather than a standalone CLI or API.

### Questions That Remain Open

| Uncertainty or assumption | How I would investigate it |
|---|---|
| Overlap may preserve evidence split between chunks, but its effect has not been measured. | Compare overlap 0, 100, and 200 with the same chunk size, questions, embedding model, and `k`. Relabel reference chunks after each rebuild. |
| BGE may be suitable, but it has not been shown to outperform alternatives on these books. | Build separate indexes and compare Hit@5, MRR@5, ingestion time, and query time using the same evidence references. |
| A threshold of 0.5 rejected one unsupported example, but it may reject useful passages for other questions. | Test more supported and unsupported questions, select a threshold on a development set, and evaluate it on held-out questions. |
| Qwen's evaluator scores may disagree with human judgments. | Manually label claim support and answer correctness, then inspect disagreements with automated scores. |
| Some repeated failures might depend on model choice rather than retrieval alone. | Give different generation models the exact same saved passages and questions, then compare answer quality and runtime. |
| Minimum memory requirements and setup portability are not established. | Test a clean environment, record hardware and peak memory, and check imports, ingestion, and sample answers on CPU and MPS. |

These are open questions or assumptions, not observed improvements.

## 4. Proposed Future Work

The following work is **not completed**. It is listed in the order I would prioritize it.

1. **Expand and stabilize evaluation.** Add factual, multi-passage, and unsupported questions across all four books. Save document hashes, evidence excerpts, settings, and outputs so comparisons remain traceable when chunk IDs change.
2. **Verify the submission from a clean environment.** Follow the README without relying on the existing database, record hardware and setup results, and preserve the model revisions used.
3. **Investigate retrieval failures.** Start with overlap experiments and inspect whether the missing reference passages move into the top five. Compare another embedding model or retrieval approach only after establishing this controlled baseline.
4. **Investigate generation and evaluator failures separately.** Use fixed retrieved context to test prompting or model changes, and manually check the evaluator's judgments.
5. **Improve ingestion reliability.** Add deterministic stored chunk IDs and completion checks, then test interrupted ingestion and document replacement.
6. **Revisit follow-up questions.** Build a test set containing both pronoun-based follow-ups and unrelated standalone questions. Measure whether rewriting preserves meaning before adding chat history back into the baseline.

For each experiment, I would keep unrelated settings fixed, report both successes and failures, and compare against the saved baseline. I would not claim an improvement until the results support it.