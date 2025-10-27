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
import raindrop.analytics as raindrop
import os
import uuid

raindrop.write_key = os.environ.get("RAINDROP_WRITE_KEY", "")
raindrop.set_debug_logs(True)
openai_api_key = os.getenv("OPENAI_API_KEY", "")

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = None
model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
pipe = transformers.pipeline("text-generation", model, dtype=dtype, device=device, )
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
create_table = 'CREATE TABLE "paper_authorships" ("conference" TEXT, "year" INTEGER, "title" TEXT, "author" TEXT, "affiliation" TEXT, PRIMARY KEY ("conference", "year", "title", "author"))'
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

def generate_sql_expensive(message, history, system_message, max_tokens):
    client = OpenAI(api_key=openai_api_key)
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            # instructions=system_message + system_prompt_tool_call,
            input=system_message + system_prompt_tool_call + f"\n\nUser query: {json.dumps({"user_query": message})}",
            reasoning={"effort": "high"},
            max_output_tokens=max_tokens,
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
        if len(response.output) == 1: # the output is [thinking, sql], so if only one item, just thinking
            return "just thinking... :("
        return response.output[-1].input
    except BadRequestError as e:
        print(f"OpenAI API BadRequestError: {e}")
        return f"-- Bad Request! {json.dumps(str(e))}"

def generate_sql_expensive_streaming(message, history, system_message):
    # TODO: figure out why streaming takes as long to return the first and final chunk with CFG grammar as the non-streaming version (much like Outlines, vs llama.cpp)
    client = OpenAI(api_key=openai_api_key)
    buffer = ""
    with client.responses.stream(
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
        ],
    ) as response_stream:
        for event in response_stream:
            if event.type == "response.output_text.delta":
                buffer += event.delta
                yield {"response": "partial", "delta": event.delta, "text": buffer}
            elif event.type == "response.completed":
                yield {"response": "complete", "text": event.response.output}
            # elif event.error:
                # yield {"response": None, "error": event.error}


def run_sql_query(sql_to_run):
    new_database = chdb.connect(":memory:").cursor()
    new_database.execute(create_table)
    new_database.execute("insert into paper_authorships select * from Python(papers)")

    try:
        new_database.execute(sql_to_run)
        results = new_database.fetchall()
        columns = [description[0] for description in new_database.description]
        result_df = pd.DataFrame(results, columns=columns)
        return f"Here are the results of `{sql_to_run}`:\n{result_df.to_string(index=False)}"
    except Exception as e:
        return f"Error executing SQL `{sql_to_run}`: {e}"


def respond(message, history, system_message, max_tokens):
    event_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    convo_id = str(uuid.uuid4())
    interaction = raindrop.begin(event_id=event_id, event="chat_message", user_id=user_id, input=message, convo_id=convo_id)
    if openai_api_key == "":
        sql_to_run = generate_sql(message, history, system_message, max_tokens)
        yield f"_(Using the SmolLM2-1.7B model, started to run `{sql_to_run}`)_\n"
        result = run_sql_query(sql_to_run)
        success = "Error executing" not in result
        interaction.add_attachments([{"type": "text", "name": "success", "value": str(success), "role": "output"}])
        interaction.finish(output=sql_to_run)
        yield run_sql_query(sql_to_run)
    else:
        sql_to_run = generate_sql_expensive(message, history, system_message, max_tokens)
        interaction.finish(output=sql_to_run)
        yield f"_(Using the GPT-5 model, started to run `{sql_to_run}`)_"
        result = run_sql_query(sql_to_run)
        success = "Error executing" not in result
        interaction.add_attachments([{"type": "text", "name": "success", "value": str(success), "role": "output"}])
        yield run_sql_query(sql_to_run)
        interaction.finish(output=sql_to_run)

demo = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value=baseline_chat_to_sql_system_prompt, label="System message"),
        gr.Slider(minimum=1, maximum=65536, value=32768, step=1, label="Max tokens"),
    ],
    title="Chat with AI Paper Titles - SQL Generator",
)

if __name__ == "__main__":
    demo.launch()
