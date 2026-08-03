# Getting started

A first-run walkthrough, from an empty app to your first cited answer.

Jump to: [1. Databases](#databases) · [2. Keys](#keys) · [3. Installs](#installs) ·
[4. Create a corpus](#corpus) · [5. Ingest](#ingest) · [6. Ask](#ask) ·
[7. Evaluate](#evaluate)

<a id="databases"></a>
## 1. Get the databases ready

Knowledge Agent keeps your knowledge base in two places:

- **Neo4j** (the knowledge graph): a separate database you run yourself and point
  the app at. Install and start it now (see [connecting Neo4j and LanceDB](#connect)
  below). You enter its address, username, and password later, when you create a
  corpus (step 4); each corpus can point at its own Neo4j database.
- **LanceDB** (the vector store): embedded, saved to a folder on disk, with no
  server to run and nothing to install. Its folder lives inside the corpus folder
  and is created on your first ingest.

You do not enter database details in Settings. **Settings → Database connection**
shows the active corpus's connection read-only. You set a corpus's Neo4j connection
once, in **Library → Create New**, when you make it.

<a id="keys"></a>
## 2. Add your API keys

Open **Keys** and paste the keys for the providers you'll use, for example an
Anthropic key for the answering model and a Voyage key for embeddings. Prefer to
run everything locally? Use **Ollama** for the answering model and the local
**HuggingFace** embedder, and you can skip the keys entirely. (Ollama covers only
the answering model; ingest still needs an embedder, so a local embedder is what
keeps it keyless.)

<a id="installs"></a>
## 3. Install what you need

Open **Installs** and install your LLM provider and an embedder, plus (optionally)
an entity extractor, a document parser, and ontologies. These are the pieces
that ingest documents and answer questions.

<a id="corpus"></a>
## 4. Create a corpus

Open **Library → Create New** and make a corpus, the container for one knowledge
base. This is where you enter the **Neo4j connection** (address, user, password) and
choose the **folder** that will hold the corpus. Then select the corpus in the
**Selected corpus** dropdown (top-right) so every tab works on it.

<a id="ingest"></a>
## 5. Ingest documents

Open **Library → Ingest**, choose a folder or files, and start. The app chunks and
embeds each document into LanceDB and records chunk nodes in Neo4j. Extracting
**entities** and **relationships** into the graph is optional: those layers are off
by default, and you turn them on in the corpus config. Larger corpora take longer;
you can watch progress and cancel a folder run.

<a id="ask"></a>
## 6. Ask a question

Open **Search**, type a question, and send. The answer appears in the right column
with citations. Use the **Retrieval** and **LLMs** sub-tabs (same column) to tune
how it searches and which model answers.

<a id="evaluate"></a>
## 7. (Optional) Check quality

Open **Evaluation** to build test cases and track retrieval and answer quality
over time.

<a id="connect"></a>
## Connecting Neo4j and LanceDB

**Neo4j (the knowledge graph)** runs as a separate database. The easiest way to
get one locally:

1. Download **Neo4j Desktop** from
   [neo4j.com/download](https://neo4j.com/download/) and install it.
2. In Neo4j Desktop, create a **new local DBMS**, give it a password, and
   **start** it. It listens on `bolt://localhost:7687` by default.
3. In Knowledge Agent, open **Library → Create New** and enter that connection: the
   bolt URL, the username (`neo4j` by default), and the password you set. The URL
   field is pre-filled with the equivalent `neo4j://127.0.0.1:7687`. The connection
   is stored per corpus and set at creation; **Settings → Database connection**
   later shows it read-only.

The app talks to Neo4j over the **bolt** protocol using those credentials; on
launch it bridges the active corpus's stored password into the environment so
graph reads work immediately. Each corpus can point at its own Neo4j database.

**LanceDB (the vector store)** needs no download and no server: it's an *embedded*
store saved to a folder. You don't set its path directly. It lives inside the
corpus folder (at `lancedb/`) and is created on your first ingest. That's the split:
the vector store lives inside the corpus folder, the graph lives in Neo4j.

**Both run locally by default.** Neo4j is a database *process* on your own machine
(`localhost:7687`), so "connecting" just means pointing the app at that local
address with your username and password. You *can* target a remote Neo4j server by
changing the bolt URL, but local is the simplest. LanceDB has no host at all: it's
embedded, just a folder of files the app opens directly, so there's nothing to run
or connect to. In the default setup neither database is a cloud service: your
documents and graph stay on your machine.
