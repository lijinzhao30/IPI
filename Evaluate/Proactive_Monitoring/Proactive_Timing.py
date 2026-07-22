import json
import sys
from typing import Dict, List
import argparse
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULT_FILE_PATH = os.path.join(PROJECT_ROOT, "Result", "Qwen3_VL_8B", "Proactive_Monitoring", "Proactive_Timing.json")
BENCHMARK_FILE_PATH = os.path.join(PROJECT_ROOT, "Benchmark", "Proactive_Monitoring", "Proactive_Timing.json")
RESULT_JSON_PATH = RESULT_FILE_PATH
BENCHMARK_JSON_PATH = BENCHMARK_FILE_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL-8B results")
    parser.add_argument("--result", default=RESULT_FILE_PATH, help="Path to the result JSON file")
    parser.add_argument("--benchmark", default=BENCHMARK_FILE_PATH, help="Path to the benchmark JSON file")
    return parser.parse_args()

TASK_TYPES = ["Action", "Event", "Object", "Text"]

def load_json(file_path: str, file_label: str) -> List[dict]:
    if not file_path:
        print(f"Error: {file_label} path is empty. Please pass it with --result or --benchmark.")
        sys.exit(1)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Cannot find {file_label} {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: {file_label} {file_path} has invalid JSON format")
        sys.exit(1)

def get_result_file_path() -> str:
    return RESULT_FILE_PATH

def extract_task_type(task: str) -> str:
    if not isinstance(task, str) or '/' not in task:
        return ""
    return task.rsplit('/', 1)[-1].strip()

def is_time_correct(result_time: int, answers: List[dict]) -> bool:
    if result_time == -2:
        return False

    if not answers or not isinstance(answers, list):
        return False

    trigger_time = answers[0].get('trigger_time')
    if trigger_time is None:
        return False

    return trigger_time - 1 <= result_time <= trigger_time + 1

def main():
    global RESULT_FILE_PATH, BENCHMARK_FILE_PATH, RESULT_JSON_PATH, BENCHMARK_JSON_PATH
    args = parse_args()
    RESULT_FILE_PATH = RESULT_JSON_PATH = args.result
    BENCHMARK_FILE_PATH = BENCHMARK_JSON_PATH = args.benchmark
    result_file_path = get_result_file_path()
    result_data = load_json(result_file_path, "result file")
    benchmark_data = load_json(BENCHMARK_FILE_PATH, "benchmark file")

    if len(result_data) != len(benchmark_data):
        print(
            f"Error: Sample count mismatch. Result count: {len(result_data)}, Benchmark count: {len(benchmark_data)}"
        )
        sys.exit(1)

    benchmark_dict: Dict[int, dict] = {}
    for index, item in enumerate(benchmark_data):
        item_id = item.get('id', index)
        benchmark_dict[item_id] = item

    result_ids = {item.get('id', index) for index, item in enumerate(result_data)}
    benchmark_ids = set(benchmark_dict.keys())
    if result_ids != benchmark_ids:
        missing_in_result = sorted(benchmark_ids - result_ids)
        missing_in_benchmark = sorted(result_ids - benchmark_ids)
        print("Error: Result and benchmark sample IDs do not match.")
        if missing_in_result:
            print(f"Benchmark IDs missing in result: {missing_in_result[:20]}")
        if missing_in_benchmark:
            print(f"Result IDs missing in benchmark: {missing_in_benchmark[:20]}")
        sys.exit(1)

    stats: Dict[str, Dict[str, int]] = {
        task_type: {"correct": 0, "total": 0} for task_type in TASK_TYPES
    }

    for index, result_item in enumerate(result_data):
        item_id = result_item.get('id', index)
        benchmark_item = benchmark_dict[item_id]

        result_time = result_item.get('result_time')
        if result_time == -2:
            print(f"Error: Sample ID {item_id} has result_time -2. Exiting.")
            sys.exit(1)

        task = benchmark_item.get('task', result_item.get('task', ''))
        task_type = extract_task_type(task)
        if task_type not in stats:
            print(f"Error: Sample ID {item_id} has unrecognized task type: {task}")
            sys.exit(1)

        stats[task_type]["total"] += 1

        answers = result_item.get('answer')
        if not answers:
            answers = benchmark_item.get('answer', [])

        if is_time_correct(result_time, answers):
            stats[task_type]["correct"] += 1

    accuracies: Dict[str, float] = {}
    for task_type in TASK_TYPES:
        total = stats[task_type]["total"]
        correct = stats[task_type]["correct"]
        accuracies[task_type] = correct / total if total > 0 else 0.0

    final_score = sum(accuracies[task_type] for task_type in TASK_TYPES) / 4

    print(f"result file: {result_file_path}")
    print(f"benchmark file: {BENCHMARK_FILE_PATH}")
    print("=" * 40)
    for task_type in TASK_TYPES:
        print(f"{task_type}:")
        print(f"  Total samples: {stats[task_type]['total']}")
        print(f"  Correct samples: {stats[task_type]['correct']}")
        print(f"  Accuracy: {accuracies[task_type]:.2%}")
        print("-" * 40)

    print(f"Average accuracy over four types: {final_score:.2%}")

if __name__ == "__main__":
    main()
