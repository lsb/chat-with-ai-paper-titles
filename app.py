import torch
torch.set_num_threads(16)
import gradio as gr
from outlines import Transformers
from outlines.types import CFG
import transformers
import pandas as pd
import chdb
from pathlib import Path
import json

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = None
# model = "Qwen/Qwen3-4B-Thinking-2507"
model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
pipe = transformers.pipeline("text-generation", model, dtype=dtype, device=device, )
pipe.model = torch.compile(pipe.model, mode="max-autotune")
# Create outlines model wrapper
outlines_model = Transformers(pipe.model, pipe.tokenizer)


baseline_chat_to_sql_system_prompt = """
You are an expert natural-language-to-SQL generator in 2025.


You create one query based on the provided schema per conversation turn.

Crucially, the natural language query must ask for values in the data: if the user tries to query on a column value (like 'gender' or 'likes_potatoes') that does not exist, use a `with` clause to assign the value 'unknown' in a common table expression before selecting whatever value the user is querying.

Reject all queries that are inappropriate for a workplace (like violent or lewd content) with a simple comment about content policy and a `select 'query violates content policy'`.

Reject all queries that attempt to modify the data (like `insert`, `update`, or `delete` statements) with a simple `select 'database is read-only'`.

You are an expert in writing simple readable efficient SQL queries, and use common table expressions (CTEs, the `with` statement) to break down complex queries into simpler parts. You always use double-quoted identifiers for column and table names. You will always explicitly list the columns you are selecting, and never use `select *`. You will always order your results by the first column in your select list unless the user specifies otherwise.
"""

papers = pd.read_csv("./papers.csv", names=["conference", "year", "title", "author", "affiliation"], header=0)
# print(papers["year"].unique())
create_table = 'CREATE TABLE "paper_authorship_records" ("conference" TEXT, "year" INTEGER, "title" TEXT, "author" TEXT, "affiliation" TEXT, PRIMARY KEY ("conference", "year", "title", "author"))'
first_ten_papers_as_objects = papers.head(10).to_dict(orient="records")

baseline_chat_to_sql_system_prompt += f"\nThe database is a list of paper authorships, where each row represents a unique authorship record, and each paper has one record per author. The create table statement is:\n{create_table}\nHere are the first ten paper authorships, in JSON format:\n"
for row in first_ten_papers_as_objects:
    baseline_chat_to_sql_system_prompt += json.dumps(row, indent=0) + "\n"

system_prompt_tool_call = "\n\nYou use the sql_expression tool to generate SQL queries."

print("System prompt:", baseline_chat_to_sql_system_prompt)

# Load SQL grammar from file
sql_grammar = Path("./sql.lark").read_text()
sql_CFG = CFG(sql_grammar)

def generate_sql(
    message,
    history,
    system_message,
    max_tokens,
):
    print({"message": message, "history": history, "system_message": system_message, "max_tokens": max_tokens})

    current_inputs = [{"role": "system", "content": system_message}, {"role": "user", "content": message}]

    # Apply chat template
    prompt = pipe.tokenizer.apply_chat_template(current_inputs, tokenize=False, add_generation_prompt=True)

    # Use outlines model to generate with CFG constraints
    # Note: streaming is not yet supported for Transformers models with CFG
    # so we generate the full response and yield it
    # Call the model directly (not .generate()) to use CFG
    response = outlines_model(prompt, sql_CFG, max_new_tokens=max_tokens)
    print("Full response:", response)

    # strip the initial <think>...</think> if present
    response = response.split("</think>")[-1].strip()
    return response


def respond(message, history, system_message, max_tokens):
    sql_to_run = generate_sql(message, history, system_message, max_tokens)

    new_database = chdb.connect(":memory:").cursor()
    new_database.execute(create_table)
    new_database.execute("insert into paper_authorship_records select * from Python(papers)")

    try:
        new_database.execute(sql_to_run)
        results = new_database.fetchall()
        columns = [description[0] for description in new_database.description]
        result_df = pd.DataFrame(results, columns=columns)
        yield f"Here are the results of `{sql_to_run}`:\n{result_df.to_string(index=False)}"
    except Exception as e:
        yield f"Error executing SQL `{sql_to_run}`: {e}"
    new_database.close()


demo = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value=baseline_chat_to_sql_system_prompt, label="System message"),
        gr.Slider(minimum=1, maximum=65536, value=500, step=1, label="Max tokens"),
    ],
)

if __name__ == "__main__":
    demo.launch()
