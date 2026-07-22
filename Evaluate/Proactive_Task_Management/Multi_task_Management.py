import json
import sys
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULT_FILE_PATH = os.path.join(PROJECT_ROOT, "Result", "Qwen3_VL_8B", "Proactive_Task_Management", "Multi_task_Management.json")
BENCHMARK_FILE_PATH = os.path.join(PROJECT_ROOT, "Benchmark", "Proactive_Task_Management", "Multi_task_Management.json")
RESULT_JSON_PATH = RESULT_FILE_PATH
BENCHMARK_JSON_PATH = BENCHMARK_FILE_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL-8B results")
    parser.add_argument("--result", default=RESULT_FILE_PATH, help="Path to the result JSON file")
    parser.add_argument("--benchmark", default=BENCHMARK_FILE_PATH, help="Path to the benchmark JSON file")
    return parser.parse_args()

def main():
    global RESULT_FILE_PATH, BENCHMARK_FILE_PATH, RESULT_JSON_PATH, BENCHMARK_JSON_PATH
    args = parse_args()
    RESULT_FILE_PATH = RESULT_JSON_PATH = args.result
    BENCHMARK_FILE_PATH = BENCHMARK_JSON_PATH = args.benchmark
    try:
        with open(RESULT_JSON_PATH, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
    except Exception as e:
        print(f"Error reading result file: {e}")
        sys.exit(1)

    try:
        with open(BENCHMARK_JSON_PATH, 'r', encoding='utf-8') as f:
            bench_data = json.load(f)
    except Exception as e:
        print(f"Error reading benchmark file: {e}")
        sys.exit(1)

    for item in result_data:
        result_time = item.get('result_time', [])
        if -2 in result_time:
            print(f"Error: Found sample with -2 in result_time (id: {item.get('id', 'unknown')}). Exiting.")
            sys.exit(1)

    if len(result_data) != len(bench_data):
        print(f"Error: Sample counts do not match! Result file has {len(result_data)} samples, benchmark file has {len(bench_data)} samples. Exiting.")
        sys.exit(1)

    bench_dict = {item['id']: item for item in bench_data if 'id' in item}

    total_samples = len(result_data)
    correct_count = 0

    for res_item in result_data:
        res_id = res_item.get('id')
        if res_id not in bench_dict:
            print(f"Warning: Result id {res_id} not found in benchmark. Skipping.")
            continue
            
        bench_item = bench_dict[res_id]
        
        result_time = res_item.get('result_time', [])
        try:
            answer = bench_item['answer']
            trigger_time_0 = answer[0]['trigger_time']
            trigger_time_1 = answer[1]['trigger_time']
        except (KeyError, IndexError):
            print(f"Warning: Missing answer or trigger_time for id {res_id}. Skipping.")
            continue

        if len(result_time) < 2:
            print(f"Warning: result_time for id {res_id} has less than 2 elements. Skipping.")
            continue

        if (trigger_time_0 - 1 <= result_time[0] <= trigger_time_0 + 1) and \
           (trigger_time_1 - 1 <= result_time[1] <= trigger_time_1 + 1):
            correct_count += 1

    print("\n" + "="*50)
    print(f"Multi-Task Evaluation Results ({total_samples} samples)")
    print("="*50)
    print(f"Correct Rate: {correct_count}/{total_samples} ({correct_count/total_samples*100:.2f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
