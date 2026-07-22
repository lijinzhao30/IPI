import json
import sys
import argparse
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULT_FILE_PATH = os.path.join(PROJECT_ROOT, "Result", "Qwen3_VL_8B", "Proactive_Monitoring", "Repeated_Proactiveness.json")
BENCHMARK_FILE_PATH = os.path.join(PROJECT_ROOT, "Benchmark", "Proactive_Monitoring", "Repeated_Proactiveness.json")
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
        with open(RESULT_FILE_PATH, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Cannot find result file {RESULT_FILE_PATH}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: result file {RESULT_FILE_PATH} has invalid JSON format")
        sys.exit(1)

    try:
        with open(BENCHMARK_FILE_PATH, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Cannot find benchmark file {BENCHMARK_FILE_PATH}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: benchmark file {BENCHMARK_FILE_PATH} has invalid JSON format")
        sys.exit(1)

    if len(result_data) != len(benchmark_data):
        print(f"Error: Sample count mismatch. Result count: {len(result_data)}, Benchmark count: {len(benchmark_data)}")
        sys.exit(1)

    correct_samples = 0
    total_samples = len(result_data)

    for idx, item in enumerate(result_data):
        result_time_list = item.get('result_time', [])
        
        if not isinstance(result_time_list, list):
            print(f"Error: Index {idx} has a non-list result_time.")
            sys.exit(1)
            
        if -2 in result_time_list:
            print(f"Error: Index {idx} has -2 in result_time. Exiting.")
            sys.exit(1)
            
        answers = item.get('answer', [])
        
        if len(result_time_list) != len(answers):
            continue
            
        is_sample_correct = True
        for i in range(len(result_time_list)):
            r_time = result_time_list[i]
            trigger_time = answers[i].get('trigger_time')
            
            if trigger_time is None:
                is_sample_correct = False
                break
                
            if not (trigger_time - 1 <= r_time <= trigger_time + 1):
                is_sample_correct = False
                break
                
        if is_sample_correct:
            correct_samples += 1

    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    print(f"Total samples: {total_samples}")
    print(f"Correct samples: {correct_samples}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
