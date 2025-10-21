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
from openai import OpenAI, BadRequestError
from concurrent.futures import ThreadPoolExecutor

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


You create one simple straightforward query based on the provided schema per conversation turn.

Crucially, the natural language query must ask for values in the data: if the user tries to query on a column value (like 'gender' or 'likes_potatoes') that does not exist, use a `with` clause to assign the value 'unknown' in a common table expression before selecting whatever value the user is querying.

Reject all queries that are inappropriate for a workplace (like violent or lewd content) with a simple comment about content policy and a `select 'query violates content policy'`.

Reject all queries that attempt to modify the data (like `insert`, `update`, or `delete` statements) with a simple `select 'database is read-only'`.

You are an expert in writing simple readable SQL queries, and use common table expressions (CTEs, the `with` statement) to break down complex queries into simpler parts. You always use double-quoted identifiers for column and table names. You will always explicitly list the columns you are selecting, and never use `select *`. You will always order your results by the first column in your select list unless the user specifies otherwise.
"""

papers = pd.read_csv("./papers.csv", names=["conference", "year", "title", "author", "affiliation"], header=0)
# print(papers["year"].unique())
create_table = 'CREATE TABLE "paper_authorship_records" ("conference" TEXT, "year" INTEGER, "title" TEXT, "author" TEXT, "affiliation" TEXT, PRIMARY KEY ("conference", "year", "title", "author"))'
# find ten randomly chosen papers
ten_papers_as_objects = papers.sample(n=10, random_state=42).to_dict(orient="records")

baseline_chat_to_sql_system_prompt += f"\nThe database is a list of paper authorships, where each row represents a unique authorship record, and each paper has one record per author. The create table statement is:\n{create_table}\nHere are the first ten paper authorships, in JSON format:\n"
for row in ten_papers_as_objects:
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

    return response

def generate_sql_expensive(message, history, system_message, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            # instructions=system_message + system_prompt_tool_call,
            input=system_message + system_prompt_tool_call + f"\n\nUser query: {json.dumps({"user_query": message})}",
            reasoning={"effort": "high"},
        max_output_tokens=10000,  # maybe we need to think a lot?
        tools=[
            {
                "type": "custom",
                "name": "sql_expression",
                "description": "Generates a SQL query based on the provided database schema and user question as per content guidelines.",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": sql_grammar,
                },
            },
        ]
        )
        print(response.output)
        if len(response.output) == 1:
            return "" # the only thing GPT did was thinking for 10000 tokens
        return response.output[-1].input
    except BadRequestError as e:
        print(f"OpenAI API BadRequestError: {e}")
        return f"-- Bad Request! {json.dumps(str(e))}"


def run_sql_query(sql_to_run):
    new_database = chdb.connect(":memory:").cursor()
    new_database.execute(create_table)
    new_database.execute("insert into paper_authorship_records select * from Python(papers)")

    try:
        new_database.execute(sql_to_run)
        results = new_database.fetchall()
        columns = [description[0] for description in new_database.description]
        result_df = pd.DataFrame(results, columns=columns)
        return f"Here are the results of `{sql_to_run}`:\n{result_df.to_string(index=False)}"
    except Exception as e:
        return f"Error executing SQL `{sql_to_run}`: {e}"


def respond(message, history, system_message, max_tokens, openai_api_key):
    if openai_api_key == "":
        sql_to_run = generate_sql(message, history, system_message, max_tokens)
        yield "Using a small open-source model (please enter your openai api key below):" + run_sql_query(sql_to_run)
    else:
        result = "Using OpenAI API, with a small open-source model as fallback..."
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_sql_expensive = executor.submit(generate_sql_expensive, message, history, system_message, openai_api_key)
            future_sql = executor.submit(generate_sql, message, history, system_message, max_tokens)
            try:
                sql = future_sql.result(timeout=120) # we run faster than 1 tok/s
                result += "\n\n*Fast model*\n" + run_sql_query(sql)
                yield result
            except TimeoutError:
                result += "\n\n*Fast model timed out after two minutes*\n"
                yield result
            try:
                sql_expensive = future_sql_expensive.result(timeout=300) # give it up to 5 minutes
                result += "\n\n*Expensive model*\n" + run_sql_query(sql_expensive)
                yield result
            except TimeoutError:
                result += "\n\n*Expensive model timed out after five minutes*\n"
                yield result


demo = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value=baseline_chat_to_sql_system_prompt, label="System message"),
        gr.Slider(minimum=1, maximum=65536, value=100, step=1, label="Max tokens"),
        gr.Textbox(value="", label="OpenAI API Key", type="password"),
    ],
)

if __name__ == "__main__":
    demo.launch()
