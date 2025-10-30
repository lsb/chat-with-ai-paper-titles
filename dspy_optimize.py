import dspy
from app import baseline_chat_to_sql_system_prompt, papers, run_sql_query, sql_grammar
from pathlib import Path
import os
import sqlite3
import pandas as pd
import json
from tqdm import tqdm
import random

def sql_query_df(sql_to_run): # using sqlite, for correctness. Runs all 883 training examples in about 200 seconds
    new_database = sqlite3.connect(":memory:")
    papers.to_sql("paper_authorships", new_database, index=False)
    new_database.execute("create index idx_year on paper_authorships(year);")
    new_database.execute("analyze;")
    try:
        # make a cursor and fetch all the results
        cursor = new_database.cursor()
        cursor.execute(sql_to_run)
        results = cursor.fetchall()
        result_df = pd.DataFrame(results)
        return result_df
    except Exception as e:
        return pd.DataFrame({"error": [f"Error executing SQL `{sql_to_run}`: {e}"]})


model = "openai/gpt-5"
not_so_smol_model = "openai/gpt-5-mini"
smol_model = "ollama_chat/qwen34bcpu"
# smol_model = "ollama_chat/qwen3:4b-thinking-2507-q8_0"
# smol_model = "ollama_chat/smollm217b-cpu"
# smol_model = "ollama_chat/smollm2:1.7b"

if os.getenv("NO_OLLAMA"):
    refinement_lm = dspy.LM(model=model, temperature=1.0, api_key=os.getenv("OPENAI_API_KEY"), max_tokens=50000, cache=False)
    lm = dspy.LM(model=not_so_smol_model, temperature=1.0, api_key=os.getenv("OPENAI_API_KEY"), max_tokens=50000, cache=False,
        # tools=[
        #     {
        #         "type": "custom",
        #         "name": "sql_expression",
        #         "description": "Generates a SQL query based on the provided database schema and user question as per content guidelines.",
        #         "format": {
        #             "type": "grammar",
        #             "syntax": "lark",
        #             "definition": sql_grammar,
        #         },
        #     }
        # ]
    )
else:
    lm = dspy.LM(model=smol_model, temperature=1.0, api_base="http://localhost:11434", api_key="", max_tokens=50000, cache=False)
    # refinement_lm = dspy.LM(model=model, temperature=1.0, api_base="http://localhost:11434", api_key="", max_tokens=32000, cache=False)
    refinement_lm = dspy.LM(model=model, temperature=1.0, api_key=os.getenv("OPENAI_API_KEY"), max_tokens=32000, cache=False)

dspy.configure(lm=lm) #, adapter=dspy.ChatAdapter(use_native_function_calling=True))

training_data_lines = [json.loads(line) for line in tqdm(Path("./training_data.jsonl").read_text().split("\n"))]
# shuffle training_data_lines with a fixed seed
random.seed(42)
random.shuffle(training_data_lines)
training_data = []
for line in tqdm(training_data_lines):
    query = line['query']
    answer = line['sql']
    print(f"Loading query {query} with answer {answer}")
    solution = "we'll compute it later!" # sql_query_rows(answer) # TODO: automate manual data cleaning
    if 'Error executing' not in solution:  # Check if the solution is valid
        training_data.append(
            dspy.Example(
                {"problem": query.strip(), "answer": answer.strip()}
            ).with_inputs("problem"))

print(f"Loaded {len(training_data)} training examples. Last five:", training_data[-5:])


class GenerateResponse(dspy.Signature):
    """replaced"""
    problem = dspy.InputField()
    answer = dspy.OutputField()

GenerateResponse.__doc__ = baseline_chat_to_sql_system_prompt

program = dspy.ChainOfThought(GenerateResponse)

def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = example['answer'].strip().lower()
    print(prediction)
    predicted_answer = prediction.answer.split("</think>")[-1].strip().lower()
    if correct_answer == predicted_answer:
        print("🎉")
        return dspy.Prediction(score=1, feedback="100% Correct")
    correct_runtime_result = sql_query_df(correct_answer)
    predicted_runtime_result = sql_query_df(predicted_answer)
    set_of_all_the_column_values_of_a_dataframe = lambda df: frozenset(frozenset(r.values()) for r in df.to_dict(orient="records"))
    if 'error' in predicted_runtime_result.columns:
        print("🙈")
        return dspy.Prediction(score=0.0, feedback=predicted_runtime_result['error'].iloc[0])
    if set_of_all_the_column_values_of_a_dataframe(correct_runtime_result) == set_of_all_the_column_values_of_a_dataframe(predicted_runtime_result):
        print(correct_runtime_result)
        print(predicted_runtime_result)
        print("✅")
        return dspy.Prediction(score=0.999, feedback="Semantically Correct")
    return dspy.Prediction(score=0.0, feedback="Incorrect SQL Result")

def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return metric_with_feedback(example, prediction, trace, pred_name, pred_trace).score

evaluate = dspy.Evaluate(
    devset=training_data,
    metric=metric,
    num_threads=10, # the endpoint can do 500k tokens per minute and this can overload it
    display_table=True,
    display_progress=True,
)

if not os.environ.get("NO_EVALUATE_FIRST"):
    evaluate(program)

optimizer = dspy.GEPA(
    metric=metric_with_feedback,
    # auto="heavy",
    max_full_evals=18, # heavy is 18
    num_threads=4,
    track_stats=True,
    reflection_minibatch_size=3, # default is 3
    reflection_lm=refinement_lm,
)

optimized_program = optimizer.compile(
    program,
    trainset = training_data[100:],
    valset = training_data[:100],
)

print("all done optimizing!")

print(optimized_program)
evaluate(optimized_program)
print(optimized_program.predict.signature.instructions)


print("all done evaluating!")
print(optimized_program.predict.signature.instructions)



