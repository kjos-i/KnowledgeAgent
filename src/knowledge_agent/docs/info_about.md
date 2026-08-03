# What is Knowledge Agent?

Knowledge Agent is a desktop app for building a searchable **knowledge base** from
your own documents and asking questions of it in plain language.

It uses **hybrid retrieval**, two complementary stores working together:

- **Vector search (LanceDB):** your documents are split into chunks and embedded,
  so a question finds passages that are *semantically* similar even when the
  wording differs.
- **Knowledge graph (Neo4j):** entities and the relationships between them are
  extracted into a graph, so the app can answer questions that depend on
  *connections* (who relates to what, across documents).

When you ask a question, the app retrieves the most relevant chunks and graph
facts, then a language model synthesises an answer with **citations** back to the
sources.

## What you can do

- **Build corpora** from PDFs and other documents (see *Library*).
- **Ask questions** and get cited, synthesised answers (see *Search*).
- **Tune retrieval** per query: how many results, hybrid vs vector-only, MMR
  diversity re-ranking, or a direct Cypher query (see *Search → Retrieval / LLMs*).
- **Measure quality** with test cases and metrics (see *Evaluation*).
- **Bring your own models:** Anthropic, Voyage, Ollama (local), Hugging Face and
  others (see *Installs* and *Keys*).

## General-purpose

Knowledge Agent is **domain-agnostic**. It ships a broad tag taxonomy and works
across many fields rather than one: you point it at your documents and it adapts
to them.

## License & citation

Knowledge Agent is released under the **PolyForm Noncommercial License 1.0.0**.
See the `LICENSE` file in the project for the full text. In short: free for
**noncommercial use** (research, personal study, education, and other
nonprofit/public-interest purposes), including modifying and sharing under the
same terms; **commercial use requires a separate license** from the copyright
holder.

If you use Knowledge Agent in your work, a citation is appreciated:

> Kjos, I. (2026). *Knowledge Agent* (Version 0.1.0) [Computer software].
> ORCID: [0000-0002-9166-3074](https://orcid.org/0000-0002-9166-3074)

(Add a DOI here once the project has a tagged release.)
