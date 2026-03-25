# RetailRAG Finetune Assistant
### A hands-on project to learn LLMs from the ground up — Transformers → Fine-Tuning → RAG → Chatbot

This project is a learning journey. Each phase teaches one core LLM concept and has you implement it directly in this codebase. By the end, you will have a working retail shopping chatbot powered by RAG, a fine-tuned FLAN-T5 model, and a deep understanding of how modern LLMs work.

---

## Project Structure

```
RetailRAG-Finetune-Assistant/
├── app/
│   ├── rag_pipeline.py     # Document loading + chunking (done)
│   └── llm_app.py          # Streamlit chatbot app (Phase 8)
├── data/
│   └── kb/
│       ├── apparel_products.txt
│       └── paper_products.txt
├── models/
│   └── peft/full/          # Fine-tuned model saved here (Phase 6)
├── notebooks/
│   └── llm_labs.ipynb      # Your lab notebook — implement each phase here
├── vectorstore/            # ChromaDB persisted here (Phase 7)
└── requirements.txt
```

---

## Learning Roadmap

---

### Phase 1 — Foundations: How LLMs Came to Be
**Concepts:** RNN → LSTM → Attention Mechanism → Transformers

Before touching a single LLM, you need to understand *why* Transformers were invented. This phase traces the evolution of sequence models.

**What to study (reference notebook cells 4–8):**
- **RNN** — processes sequences step by step, but forgets long-range context
- **LSTM** — adds a "memory cell" to fix RNNs' vanishing gradient problem
- **Attention Mechanism** — lets the model focus on *relevant* parts of the input, not just the last hidden state
- **Transformers** — replace recurrence entirely with self-attention; run in parallel; scale massively

**Key insight to internalize:**
> Transformers process the entire sequence at once using attention scores. Every token "looks at" every other token and decides how much weight to give it. This is why they scale — no sequential bottleneck.

**Lab task (notebooks/llm_labs.ipynb):**
- Add a markdown cell explaining each concept in your own words
- Draw (or describe in text) the Transformer encoder-decoder architecture
- No code required for this phase — conceptual understanding first

**Status:** [ ] Complete

---

### Phase 2 — NLP Fundamentals: Tokenizers and Embeddings
**Concepts:** Tokenization, Subword encoding (BPE/WordPiece), Word Embeddings

**What to study (reference notebook cells 9–10):**
- **Tokenizer** — converts raw text into token IDs the model can process. Models don't see words, they see integers.
- **Subword tokenization** — "unhappiness" → ["un", "##happiness"] so the model handles unseen words
- **Embeddings** — each token ID maps to a dense vector (e.g., 768 dimensions). Similar meanings → close vectors.

**Key insight to internalize:**
> The embedding layer is a lookup table. Token ID 4821 maps to a 768-float vector. These vectors are *learned* during training — that's where meaning is encoded.

**Lab task (notebooks/llm_labs.ipynb):**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

text = "Do you have adult t-shirts in size XL?"
tokens = tokenizer(text)
print(tokens)                          # input_ids, attention_mask
print(tokenizer.convert_ids_to_tokens(tokens["input_ids"]))  # see actual tokens
```

**What to observe:**
- How many tokens does your sentence produce?
- How does the tokenizer handle "t-shirts" or "XL"?
- What is `attention_mask` and why is every value 1 here?

**Status:** [ ] Complete

---

### Phase 3 — Text Generation Strategies
**Concepts:** Greedy Search, Beam Search, Sampling, Top-K, Top-P (Nucleus Sampling)

The model outputs a probability distribution over its vocabulary at each step. *How you pick the next token* determines the quality and diversity of the output.

**What to study (reference notebook cells 11–60):**

| Strategy | How it works | Trade-off |
|---|---|---|
| Greedy | Always pick the highest-probability token | Fast but repetitive |
| Beam Search | Keep top-K sequences at each step | Better quality, slower |
| Sampling | Sample from the distribution randomly | Creative but can go off-topic |
| Top-K | Sample only from top K tokens | Cuts off the long tail |
| Top-P (Nucleus) | Sample from the smallest set of tokens whose cumulative prob ≥ P | Adapts vocabulary size dynamically |

**Key insight to internalize:**
> `temperature` controls the sharpness of the distribution. Low temp (0.1) = near-greedy. High temp (1.5) = very random. Top-P at 0.9 is the most commonly used strategy in production.

**Lab task (notebooks/llm_labs.ipynb):**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

prompt = "Summarize: The customer asked about t-shirt sizes available in navy blue."
inputs = tokenizer(prompt, return_tensors="pt")

# Try each strategy and compare outputs
greedy = model.generate(**inputs)
beam   = model.generate(**inputs, num_beams=5, early_stopping=True)
sample = model.generate(**inputs, do_sample=True, temperature=0.9, top_p=0.9)

for label, output in [("Greedy", greedy), ("Beam", beam), ("Sample", sample)]:
    print(f"{label}: {tokenizer.decode(output[0], skip_special_tokens=True)}")
```

**Status:** [ ] Complete

---

### Phase 4 — Prompting FLAN-T5: Zero / One / Few-Shot Inference
**Concepts:** Instruction-tuned models, Prompt Engineering, In-Context Learning

FLAN-T5 is a T5 model fine-tuned on hundreds of tasks using instructions. You don't need to re-train it — you steer it with the right prompt.

**What to study (reference notebook cells 61–91):**
- **Zero-shot** — give the model a task description and input, no examples
- **One-shot** — include one worked example in the prompt before your question
- **Few-shot** — include 2–5 examples; model picks up the pattern

**Key insight to internalize:**
> FLAN-T5 was trained with explicit instruction prefixes like "Summarize:", "Answer:", "Classify:". These prefixes are what make zero-shot work. Without them, the model doesn't know what task you want.

**Lab task (notebooks/llm_labs.ipynb):**

Apply this to your retail use case — product question answering:

```python
# Zero-shot
zero_shot_prompt = """Answer the customer question using only the provided product info.

Product info: Adult T-Shirts available in sizes S, M, L, XL, XXL. Colors: white, black, navy.

Question: Do you carry adult t-shirts in navy blue?
Answer:"""

# One-shot (show the model the format you want)
one_shot_prompt = """Answer the customer question using only the provided product info.

Product info: Kids T-Shirts available in sizes S, M, L.
Question: What sizes do kids t-shirts come in?
Answer: Kids t-shirts are available in Small, Medium, and Large.

Product info: Adult T-Shirts available in sizes S, M, L, XL, XXL. Colors: white, black, navy.
Question: Do you carry adult t-shirts in navy blue?
Answer:"""

for label, prompt in [("Zero-shot", zero_shot_prompt), ("One-shot", one_shot_prompt)]:
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    print(f"\n{label}:\n{tokenizer.decode(output[0], skip_special_tokens=True)}")
```

**What to observe:** How much does adding an example change the answer format and quality?

**Status:** [ ] Complete

---

### Phase 5 — Full Fine-Tuning with ROUGE Evaluation
**Concepts:** Supervised fine-tuning, dataset preprocessing, training loop, ROUGE metrics

Full fine-tuning updates *all* model parameters on your task-specific dataset. Expensive but gives maximum performance.

**What to study (reference notebook cells 92–122):**
- Dataset: `samsum` — dialogue summarization (conversation → summary)
- Preprocessing: tokenize inputs + labels, pad/truncate to fixed length
- Training: use HuggingFace `Trainer` or a manual loop
- ROUGE: measures n-gram overlap between generated and reference summaries
  - **ROUGE-1**: unigram overlap
  - **ROUGE-2**: bigram overlap
  - **ROUGE-L**: longest common subsequence

**Key insight to internalize:**
> Fine-tuning on domain data (e.g., retail conversations) will improve ROUGE scores for that domain. But full fine-tuning requires significant GPU memory because every parameter gets a gradient.

**Lab task (notebooks/llm_labs.ipynb):**
```python
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import evaluate

dataset = load_dataset("samsum", trust_remote_code=True)
print(dataset["train"][0])   # inspect a sample

# Tokenize
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    inputs  = tokenizer(examples["dialogue"], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(examples["summary"],  max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True)

# ROUGE evaluation on the base model (before fine-tuning)
rouge = evaluate.load("rouge")
# (evaluate base model predictions vs. references on a small sample)
```

**What to observe:**
- ROUGE scores of the *base* model before any fine-tuning
- After fine-tuning a few hundred steps, compare ROUGE scores — you should see improvement

**Note:** Full fine-tuning is compute-heavy. The reference project's pre-trained `full/` model is your reference output. You can load it to see what fully fine-tuned looks like.

**Status:** [ ] Complete

---

### Phase 6 — Parameter Efficient Fine-Tuning (PEFT) with LoRA
**Concepts:** LoRA, adapter layers, rank decomposition, frozen weights

Full fine-tuning updates billions of parameters. LoRA freezes the original model and adds small trainable matrices at each attention layer. Result: 43 MB of adapters vs. 2.9 GB full model — same quality.

**What to study (reference notebook cells 123–134):**
- LoRA injects two low-rank matrices **A** and **B** (rank r) into each weight matrix W
- W_new = W_frozen + B × A
- Only A and B are trained — orders of magnitude fewer parameters
- PEFT library wraps any HuggingFace model with LoRA config

**Key insight to internalize:**
> LoRA works because fine-tuning updates tend to have low intrinsic rank. A 4096×4096 weight matrix update can be approximated by two 4096×8 matrices (rank 8), saving 99.8% of parameters.

**Lab task (notebooks/llm_labs.ipynb):**
```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32,                          # rank
    lora_alpha=32,
    target_modules=["q", "v"],     # apply LoRA to query and value projections
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()   # see how few params you're training
```

**What to observe:** What percentage of parameters are trainable with LoRA vs full fine-tuning?

**Where it goes in the project:** Save the LoRA adapter to `models/peft/full/` after training.

**Status:** [ ] Complete

---

### Phase 7 — Knowledge Grounding with RAG
**Concepts:** Hallucination problem, Retrieval-Augmented Generation, Vector Databases, Embeddings, Similarity Search

LLMs hallucinate — they generate plausible-sounding but wrong answers when they don't know something. RAG fixes this by retrieving relevant facts *before* generating an answer.

**RAG pipeline:**
```
User Query → Embed Query → Search VectorDB → Retrieve Top-K Chunks → LLM(query + chunks) → Answer
```

**What to study (reference notebook cells 135–143):**
- Document loading and chunking strategy (chunk_size, chunk_overlap)
- Sentence-Transformers: embed text into dense vectors
- ChromaDB: stores vectors, supports similarity search
- LangChain QA chain: combines retriever + LLM

**Key insight to internalize:**
> The LLM never "looks up" your documents — it only sees the retrieved chunks injected into its prompt context. The retrieval step is the bottleneck for answer quality. Bad retrieval = bad answers regardless of the LLM.

**Lab task (notebooks/llm_labs.ipynb):**

This maps directly to `app/rag_pipeline.py`. Extend it:

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# Step 1: Load and chunk (already in rag_pipeline.py)
loader = DirectoryLoader("data/kb", glob="**/*.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 2: Embed and store
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="vectorstore/chroma_db")
vectordb.persist()

# Step 3: Query
query = "Do you have sweatshirts available in Europe?"
results = vectordb.similarity_search_with_score(query, k=3)
for doc, score in results:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}")
```

**What to observe:**
- What score (distance) do relevant vs. irrelevant chunks get?
- Does chunking strategy affect which chunks are retrieved?

**Add to `app/rag_pipeline.py`:** Add `build_vectorstore()` and `query_vectorstore()` functions.

**Status:** [ ] Complete

---

### Phase 8 — Streamlit Chatbot App
**Concepts:** Session state, caching, chat UI, connecting all components end-to-end

This is the final integration phase. Connect the RAG pipeline to an LLM (OpenAI GPT-3.5-Turbo or local model) and wrap it in a Streamlit web UI.

**What to study (reference `llm_app.py`):**
- `@st.cache_resource` — loads the model and vectordb once at startup
- `st.session_state` — persists chat history across reruns
- `st.chat_message` / `st.chat_input` — modern Streamlit chat UI
- Sidebar for knowledge base management (upload or use existing)
- `load_qa_chain` — LangChain chain that formats retrieved docs + question into an LLM prompt

**Lab task:** Implement `app/llm_app.py` with:
1. Sidebar: show current KB files or allow uploading new `.txt` files
2. On startup: load docs → split → embed → store in ChromaDB → initialize LLM chain
3. Chat loop: user message → retrieve relevant chunks → LLM answers → display

**Components to wire together:**
- `rag_pipeline.py`: `load_docs()`, `split_docs()` (done), + `build_vectorstore()` (Phase 7)
- `SentenceTransformerEmbeddings` + `Chroma` (Phase 7)
- `ChatOpenAI` + `load_qa_chain` (new in this phase)
- Streamlit session state for chat history

**Status:** [ ] Complete

---

## Progress Tracker

| Phase | Topic | Notebook | Project File | Status |
|-------|-------|----------|-------------|--------|
| 1 | Foundations: RNN → Transformers | llm_labs.ipynb | — | [ ] |
| 2 | Tokenizers & Embeddings | llm_labs.ipynb | — | [ ] |
| 3 | Text Generation Strategies | llm_labs.ipynb | — | [ ] |
| 4 | FLAN-T5 Zero/One/Few-Shot | llm_labs.ipynb | — | [ ] |
| 5 | Full Fine-Tuning + ROUGE | llm_labs.ipynb | models/ | [ ] |
| 6 | PEFT / LoRA | llm_labs.ipynb | models/peft/ | [ ] |
| 7 | RAG + ChromaDB | llm_labs.ipynb | app/rag_pipeline.py, vectorstore/ | [ ] |
| 8 | Streamlit Chatbot App | — | app/llm_app.py | [ ] |

---

## How to Run (after Phase 8 is complete)

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI key
export OPENAI_API_KEY="your-key-here"

# Launch the app
streamlit run app/llm_app.py
```
