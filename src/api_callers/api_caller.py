import os
import time
import json
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
import openai
import anthropic
import argparse


load_dotenv()

MODELS_CONFIG = {
    "gpt-5-mini": {
        "provider": "openai",
        "model": "gpt-5-mini",
        "type": "reasoning",
        "client": None,
        "cost_per_1M_input": 1.50,
        "cost_per_1M_output": 6.00
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "model": "gemini-2.5-flash",
        "type": "reasoning",
        "client": None,
        "cost_per_1M_input": 0.15,
        "cost_per_1M_output": 0.60
    },
    "grok-4-fast-reasoning": {
        "provider": "xai",
        "model": "grok-4-fast-reasoning",
        "type": "reasoning",
        "client": None,
        "cost_per_1M_input": 2.00,
        "cost_per_1M_output": 10.00
    },
    "claude-3-7-sonnet": {
        "provider": "anthropic",
        "model": "claude-3-7-sonnet-20250219",
        "type": "reasoning",
        "client": None,
        "cost_per_1M_input": 3.00,
        "cost_per_1M_output": 15.00
    }
}

DRY_RUN = False
NUM_OUTPUTS_PER_PROMPT = 5
MAX_PROGRAMS_TO_PROCESS = None
DELAY_BETWEEN_CALLS = 0.1
MAX_WORKERS = len(MODELS_CONFIG)

INCLUDE_DATASETS = ["CRUXEval", "HumanEval", "PythonSaga"]


def load_prompt_templates(prompt_dir: Path):
    """Load reasoning prompt templates"""
    templates = {}

    coverage_path = prompt_dir / "ask_predict_coverage.txt"
    input_path = prompt_dir / "ask_predict_input.txt"

    if coverage_path.exists():
        templates["ask_predict_coverage"] = coverage_path.read_text(encoding="utf-8")
        print(f"Loaded template: {coverage_path}")
    else:
        print(f"Missing template: {coverage_path}")

    if input_path.exists():
        templates["ask_predict_input"] = input_path.read_text(encoding="utf-8")
        print(f"Loaded template: {input_path}")
    else:
        print(f"Missing template: {input_path}")

    return templates


def setup_google_client():
    """Setup Google Gemini client"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    genai.configure(api_key=api_key)
    return genai

def setup_openai_client():
    """Setup OpenAI client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    return openai.OpenAI(api_key=api_key)

def setup_anthropic_client():
    """Setup Anthropic client"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")
    return anthropic.Anthropic(api_key=api_key)

def setup_xai_client():
    """Setup xAI client"""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY not found in environment")
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )

def initialize_clients():
    """Initialize all API clients"""
    print("Initializing API clients...")

    for model_name, config in MODELS_CONFIG.items():
        try:
            if config["provider"] == "google":
                config["client"] = setup_google_client()
            elif config["provider"] == "openai":
                config["client"] = setup_openai_client()
            elif config["provider"] == "anthropic":
                config["client"] = setup_anthropic_client()
            elif config["provider"] == "xai":
                config["client"] = setup_xai_client()
            print(f"{model_name} client initialized")
        except Exception as e:
            print(f"Failed to initialize {model_name}: {e}")
            config["client"] = None


def call_google_api(model_config, prompt_text):
    """Call Google Gemini API"""
    try:
        model = genai.GenerativeModel(model_config["model"])
        response = model.generate_content(
            prompt_text,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )

        response_text = ""
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = candidate.content.parts[0].text
            else:
                finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
                response_text = f"[NO_CONTENT - finish_reason: {finish_reason}]"
        else:
            response_text = "[NO_CANDIDATES_RETURNED]"

        usage_info = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

        if hasattr(response, 'usage_metadata'):
            usage_info["input_tokens"] = getattr(response.usage_metadata, 'prompt_token_count', 0)
            usage_info["output_tokens"] = getattr(response.usage_metadata, 'candidates_token_count', 0)
            usage_info["total_tokens"] = getattr(response.usage_metadata, 'total_token_count', 0)
        else:
            usage_info["input_tokens"] = len(prompt_text) // 4
            usage_info["output_tokens"] = len(response_text) // 4
            usage_info["total_tokens"] = usage_info["input_tokens"] + usage_info["output_tokens"]

        return response_text, usage_info

    except Exception as e:
        error_msg = f"Google API Error: {str(e)}"
        print(f"{error_msg}")
        return error_msg, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def call_openai_api(model_config, prompt_text):
    """Call OpenAI API with graceful fallbacks for unsupported params."""
    try:
        client = model_config["client"]
        base_params = {
            "model": model_config["model"],
            "messages": [{"role": "user", "content": prompt_text}],
            "max_completion_tokens": 2048
        }


        try:
            response = client.chat.completions.create(
                **base_params,
                temperature=0.7,
                top_p=0.9
            )
        except Exception as e1:
            msg = str(e1).lower()


            if "top_p" in msg and "unsupported" in msg:
                try:
                    response = client.chat.completions.create(
                        **base_params,
                        temperature=0.7
                    )
                except Exception as e2:
                    msg2 = str(e2).lower()

                    if "temperature" in msg2 and "unsupported" in msg2:
                        response = client.chat.completions.create(**base_params)
                    else:
                        raise

            elif "temperature" in msg and "unsupported" in msg:
                try:
                    response = client.chat.completions.create(
                        **base_params,
                        top_p=0.9
                    )
                except Exception as e3:
                    msg3 = str(e3).lower()
                    if "top_p" in msg3 and "unsupported" in msg3:
                        response = client.chat.completions.create(**base_params)
                    else:
                        raise
            else:
                raise

        response_text = response.choices[0].message.content
        usage_info = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        return response_text, usage_info

    except Exception as e:
        error_msg = f"OpenAI API Error: {str(e)}"
        print(f"{error_msg}")
        return error_msg, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def call_anthropic_api(model_config, prompt_text):
    """Call Anthropic API"""
    try:
        client = model_config["client"]

        message = client.messages.create(
            model=model_config["model"],
            max_tokens=2048,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt_text}]
        )

        response_text = message.content[0].text


        usage_info = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens
        }

        return response_text, usage_info

    except Exception as e:
        error_msg = f"Anthropic API Error: {str(e)}"
        print(f"{error_msg}")
        return error_msg, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def call_xai_api(model_config, prompt_text):
    """Call xAI API"""
    try:
        client = model_config["client"]

        response = client.chat.completions.create(
            model=model_config["model"],
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048
        )

        response_text = response.choices[0].message.content

        usage_info = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return response_text, usage_info

    except Exception as e:
        error_msg = f"xAI API Error: {str(e)}"
        print(f"{error_msg}")
        return error_msg, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def call_model_api(model_name, prompt_text):
    """Unified API call function with provider routing"""
    model_config = MODELS_CONFIG[model_name]

    if DRY_RUN:
        response_text = f"--- DRY RUN ({model_name}) ---\nPrompt length: {len(prompt_text)} chars"
        usage_info = {"input_tokens": len(prompt_text)//4, "output_tokens": 100, "total_tokens": len(prompt_text)//4 + 100}
        return response_text, usage_info

    try:
        if model_config["provider"] == "google":
            return call_google_api(model_config, prompt_text)
        elif model_config["provider"] == "openai":
            return call_openai_api(model_config, prompt_text)
        elif model_config["provider"] == "anthropic":
            return call_anthropic_api(model_config, prompt_text)
        elif model_config["provider"] == "xai":
            return call_xai_api(model_config, prompt_text)
        else:
            error_msg = f"Unknown provider: {model_config['provider']}"
            print(f"{error_msg}")
            return error_msg, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    except Exception as e:
        error_msg = f"API call failed for {model_name}: {str(e)}"
        print(f"{error_msg}")
        return error_msg, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def add_line_numbers(code):
    """Add line numbers to code"""
    return "\n".join(f"{i:3d} | {line}" for i, line in enumerate(code.splitlines(), 1))

def format_prompt(template, program_code, priority_line):
    """Replace placeholders with code and priority line"""
    numbered = add_line_numbers(program_code)
    formatted = template.replace("{program_code}", numbered)
    if priority_line is not None:
        formatted = formatted.replace("{priority_line}", str(priority_line))
    return formatted


def filter_programs_by_dataset(coverage_data, included_datasets):
    """Filter programs based on included datasets"""
    if not included_datasets:
        return coverage_data

    filtered_programs = []
    dataset_counts = {}

    for program in coverage_data:
        dataset = program.get("dataset", "Unknown")
        if dataset in included_datasets:
            filtered_programs.append(program)
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    print(f"Dataset filtering applied:")
    for dataset, count in dataset_counts.items():
        print(f"   {dataset}: {count} programs")

    return filtered_programs


def process_single_api_call(args):
    """Process a single API call - used for parallel execution"""
    model_name, prompt_text, output_dir, prompt_name = args

    print(f"Inspecting Calling {model_name} with prompt: {prompt_name}")


    response_text, usage_info = call_model_api(model_name, prompt_text)


    (output_dir / f"{prompt_name}_response.txt").write_text(response_text)
    (output_dir / f"{prompt_name}_usage.json").write_text(json.dumps({
        "usage_info": usage_info,
        "cost_breakdown": {
            "input_tokens": usage_info["input_tokens"],
            "output_tokens": usage_info["output_tokens"],
            "input_cost": (usage_info["input_tokens"] / 1_000_000) * MODELS_CONFIG[model_name]["cost_per_1M_input"],
            "output_cost": (usage_info["output_tokens"] / 1_000_000) * MODELS_CONFIG[model_name]["cost_per_1M_output"],
            "total_cost": (usage_info["input_tokens"] / 1_000_000) * MODELS_CONFIG[model_name]["cost_per_1M_input"] +
                         (usage_info["output_tokens"] / 1_000_000) * MODELS_CONFIG[model_name]["cost_per_1M_output"]
        }
    }, indent=2))

    print(f"{model_name} - {prompt_name} completed")

    return {
        "model_name": model_name,
        "prompt_name": prompt_name,
        "response_text": response_text,
        "usage_info": usage_info,
        "prompt_length": len(prompt_text),
        "response_length": len(response_text)
    }

def process_program_parallel(program, templates, experiment_stats, pbar, base_output_dir: Path):
    """Process a single program with all models in parallel"""
    task_id = program["task_id"]
    code = program["runnable_script"]
    priority_line = program["coverage_metadata"]["advanced_priority_line"]

    print(f"\nProcessing program: {task_id}")
    print(f"Priority line: {priority_line}")

    program_stats = {
        "task_id": task_id,
        "priority_line": priority_line,
        "models": {}
    }


    api_call_args = []

    for model_name, model_config in MODELS_CONFIG.items():
        if model_config["client"] is None and not DRY_RUN:
            print(f"Skipping {model_name} - client not initialized")
            continue


        model_dir = base_output_dir / model_name
        model_dir.mkdir(exist_ok=True, parents=True)


        for output_num in range(1, NUM_OUTPUTS_PER_PROMPT + 1):
            output_dir = model_dir / task_id.replace("/", "_") / f"output_{output_num}"
            output_dir.mkdir(exist_ok=True, parents=True)


            for prompt_name, template_text in templates.items():
                formatted_prompt = format_prompt(template_text, code, priority_line)


                (output_dir / f"{prompt_name}_prompt.txt").write_text(formatted_prompt)


                api_call_args.append((model_name, formatted_prompt, output_dir, prompt_name))

    print(f"Queueing {len(api_call_args)} API calls for {task_id}")


    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_api_call, args) for args in api_call_args]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()


                if not DRY_RUN:
                    experiment_stats["completed_calls"] += 1
                    experiment_stats["models"][result["model_name"]]["completed_calls"] += 1

                    experiment_stats["total_costs"]["input_tokens"] += result["usage_info"]["input_tokens"]
                    experiment_stats["total_costs"]["output_tokens"] += result["usage_info"]["output_tokens"]
                    experiment_stats["total_costs"]["total_tokens"] += result["usage_info"]["total_tokens"]

                    experiment_stats["models"][result["model_name"]]["tokens"]["input_tokens"] += result["usage_info"]["input_tokens"]
                    experiment_stats["models"][result["model_name"]]["tokens"]["output_tokens"] += result["usage_info"]["output_tokens"]
                    experiment_stats["models"][result["model_name"]]["tokens"]["total_tokens"] += result["usage_info"]["total_tokens"]


                    input_cost = (result["usage_info"]["input_tokens"] / 1_000_000) * MODELS_CONFIG[result["model_name"]]["cost_per_1M_input"]
                    output_cost = (result["usage_info"]["output_tokens"] / 1_000_000) * MODELS_CONFIG[result["model_name"]]["cost_per_1M_output"]
                    total_cost = input_cost + output_cost

                    experiment_stats["total_costs"]["estimated_cost"] += total_cost
                    experiment_stats["models"][result["model_name"]]["estimated_cost"] += total_cost

                pbar.update(1)
                pbar.set_postfix({
                    "Program": task_id[:20],
                    "Cost": f"${experiment_stats['total_costs']['estimated_cost']:.2f}" if not DRY_RUN else "DRY_RUN"
                })


                if not DRY_RUN:
                    time.sleep(DELAY_BETWEEN_CALLS)

            except Exception as e:
                error_msg = f"Error in parallel API call: {str(e)}"
                print(error_msg)
                if not DRY_RUN:
                    experiment_stats["failed_calls"] += 1
                pbar.update(1)

    return program_stats

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the multi-model API coverage experiment."
    )

    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=Path(
            "artifacts/programs/runner_programs_with_coverage.json"
        ),
        help="Path to the program coverage JSON file."
    )

    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path("data/prompts/reasoning"),
        help="Directory containing the reasoning prompt templates."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("API_Model_Outputs"),
        help="Directory where API outputs and statistics will be saved."
    )

    return parser.parse_args()

def main():

    args = parse_args()

    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

    print(" 4-Model API Experiment Runner ")
    print(f"Target models: {list(MODELS_CONFIG.keys())}")
    print(f"Included datasets: {INCLUDE_DATASETS}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print(f"Delay between calls: {DELAY_BETWEEN_CALLS}s")


    initialize_clients()


    ready_models = [name for name, config in MODELS_CONFIG.items() if config["client"] is not None or DRY_RUN]
    print(f"Ready models: {ready_models}")


    if not args.coverage_json.exists():
        print(f"ERROR: File '{args.coverage_json}' not found.")
        exit(1)

    try:
        with open(args.coverage_json, "r", encoding="utf-8") as f:
            coverage_data = json.load(f)
        print(f"Loaded {len(coverage_data)} programs from coverage data")
    except Exception as e:
        print(f"ERROR: Could not parse JSON: {e}")
        exit(1)


    coverage_data = filter_programs_by_dataset(coverage_data, INCLUDE_DATASETS)


    if MAX_PROGRAMS_TO_PROCESS:
        coverage_data = coverage_data[:MAX_PROGRAMS_TO_PROCESS]
        print(f"Limiting to {len(coverage_data)} programs for testing")


    templates = load_prompt_templates(args.prompt_dir)
    if not templates:
        print("No templates loaded. Exiting.")
        exit(1)


    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_programs = len(coverage_data)
    total_models = len([m for m in MODELS_CONFIG if MODELS_CONFIG[m]["client"] is not None or DRY_RUN])
    total_prompts_per_program = len(templates) * NUM_OUTPUTS_PER_PROMPT
    total_api_calls = total_programs * total_models * total_prompts_per_program

    print(f"\nExperiment Scale:")
    print(f"  Programs: {total_programs}")
    print(f"  Models: {total_models}")
    print(f"  Prompt types: {len(templates)}")
    print(f"  Outputs per prompt: {NUM_OUTPUTS_PER_PROMPT}")
    print(f"  Total API calls: {total_api_calls}")

    if DRY_RUN:
        print("\n" + "="*60)
        print("DRY RUN MODE - No real API calls")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("LIVE MODE - Real API calls will be made")
        print("="*60)
        confirm = input("Proceed? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborting script.")
            exit(0)


    experiment_stats = {
        "total_programs": total_programs,
        "total_models": total_models,
        "total_api_calls": total_api_calls,
        "completed_calls": 0,
        "failed_calls": 0,
        "models": {},
        "total_costs": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0
        },
        "config": {
            "dry_run": DRY_RUN,
            "outputs_per_prompt": NUM_OUTPUTS_PER_PROMPT,
            "max_programs": MAX_PROGRAMS_TO_PROCESS,
            "included_datasets": INCLUDE_DATASETS,
            "max_workers": MAX_WORKERS,
            "delay_between_calls": DELAY_BETWEEN_CALLS
        },
        "timing": {
            "start_time": start_time_str,
            "end_time": None,
            "duration_seconds": None
        }
    }


    for model_name in MODELS_CONFIG.keys():
        experiment_stats["models"][model_name] = {
            "completed_calls": 0,
            "failed_calls": 0,
            "tokens": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            },
            "estimated_cost": 0.0
        }


    with tqdm(total=total_api_calls, desc="Overall Progress") as pbar:
        for program in coverage_data:
            program_stats = process_program_parallel(program, templates, experiment_stats, pbar, output_dir)


    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    duration_seconds = end_time - start_time


    experiment_stats["timing"]["end_time"] = end_time_str
    experiment_stats["timing"]["duration_seconds"] = round(duration_seconds, 2)


    stats_file = output_dir / "experiment_statistics.json"
    stats_file.write_text(json.dumps(experiment_stats, indent=2))

    print("\n" + "="*60)
    print("4-MODEL API EXPERIMENT COMPLETED")
    print("="*60)
    print(f"Results saved in: {output_dir}")
    print(f"Statistics file: {stats_file}")
    print(f"Timing:")
    print(f"   Start: {start_time_str}")
    print(f"   End:   {end_time_str}")
    print(f"   Duration: {duration_seconds:.2f} seconds ({duration_seconds/60:.2f} minutes)")

    if not DRY_RUN:
        print(f"\nCOST SUMMARY:")
        print(f"   Total Input Tokens: {experiment_stats['total_costs']['input_tokens']:,}")
        print(f"   Total Output Tokens: {experiment_stats['total_costs']['output_tokens']:,}")
        print(f"   Total Tokens: {experiment_stats['total_costs']['total_tokens']:,}")
        print(f"   Estimated Total Cost: ${experiment_stats['total_costs']['estimated_cost']:.2f}")

        print(f"\nInspecting MODEL BREAKDOWN:")
        for model_name, model_stats in experiment_stats["models"].items():
            cost = model_stats["estimated_cost"]
            tokens = model_stats["tokens"]["total_tokens"]
            calls = model_stats["completed_calls"]
            print(f"   {model_name:25}: ${cost:6.2f} | {tokens:8,} tokens | {calls:3} calls")

if __name__ == "__main__":
    main()