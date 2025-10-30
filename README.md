# Chat with AI Paper Titles

A natural language to SQL interface for querying AI paper metadata using grammar-constrained LLM generation. This uses Clickhouse (in-process) and GPT-5-Mini CFG Grammar.

# Findings

## Grammar

We use a SQLite-inspired subset of SQL for the grammar.

Composing a SQL grammar that represents desirable queries is 1) supported by GPT-5 as well as open source models with Outlines; 2) rigid and error-prone; and 3) mostly redundant for modern LLMs.

SQL is in enough LLM training data that most modern LLMs will generate valid SQL without a grammar, much like JSON, although a grammar is useful to create [acrostic poetry](https://huggingface.co/spaces/lsb/acrostic-intimations). It is worth understanding why the model's prompt is inspiring invalid SQL, or inspiring invalid JSON, and optimizing the prompt.

## Clickhouse

During a DSPy prompt optimization run, in which the SQL from optimized prompts was important for steering the prompt optimization, Clickhouse reported that the table already existed on a fresh connection to an in-memory in-process `:memory:` database. This lack of thread-safety implies less isolation that a developer is likely to expect for a SQL database, and less isolation that a developer is likely to expect for an in-memory dataframe library.

Otherwise, Clickhouse was as easy to get started with as SQLite; Pandas can export a data frame to a SQL cursor, and Clickhouse in-process can directly load a Pandas dataframe into storage. When using Clickhouse in-process (via the pip package `chdb`), for safety, `run_sql_query()` creates a new database for every new query, and the load time is on the order of 10-100ms.

Because of a lack of desire to debug the aforementioned concurrency in a novel database, the DSPy optimization uses the SQLite database engine to run the SQL.

## DSPy and GEPA: Prompt Optimization via Synthetic Data from 56% correct to 95% correct

We can make a parallel corpus of natural language query on a table, and the associated SQL, and optimize the prompt to be more accurate at making the associated SQL from a natural language query (especially for avoiding invalid SQL, and for maximizing the similarity between the result sets).

We have a synthetic data prompt, we have the resulting training_data.jsonl (from `gpt-oss:20b`) with some manual cleaning and duplication of the policy violations (content and read-only) due to data imbalance.

Scoring textual equivalence as 1.0 and scoring equivalence of the resulting set of sets of each row's values as 0.999 (allowing for variation in row order and column order but all tuples must match), GEPA optimization improves the naive prompt's 484.515/883 accuracy to 840.894/883, from even odds to mostly correct.

## Evals

We have a pytest suite of a few evaluations for quick runtime checks. The bulk of the evals come from the synthetic data prompt that generated training data for DSPy.

## Data Source

The data comes from https://github.com/martenlienen/icml-neurips-iclr-dataset .

## Open source Models

Small non-thinking models, like SmolLM2 1.7B, are inaccurate and fast. Small thinking models, like Qwen3 4B Thinking 2507, reach 80-85% correct (according to the correctness criteria in the GEPA section above) with the GPT-5-Mini optimized prompt, and no grammar. Larger thinking models, like GPT-OSS 120B, are even more accurate. Outlines, which contrains grammar for the open source models, [does not work for thinking models](https://github.com/dottxt-ai/outlines/issues/1771).