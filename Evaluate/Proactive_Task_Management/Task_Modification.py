import json
import sys
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULT_FILE_PATH = os.path.join(PROJECT_ROOT, "Result", "Qwen3_VL_8B", "Proactive_Task_Management", "Task_Modification.json")
BENCHMARK_FILE_PATH = os.path.join(PROJECT_ROOT, "Benchmark", "Proactive_Task_Management", "Task_Modification.json")
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
    if not os.path.exists(RESULT_JSON_PATH):
        print(f"Error: Result JSON file not found at {RESULT_JSON_PATH}")
        sys.exit(1)
        
    if not os.path.exists(BENCHMARK_JSON_PATH):
        print(f"Error: Benchmark JSON file not found at {BENCHMARK_JSON_PATH}")
        sys.exit(1)

    with open(RESULT_JSON_PATH, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
        
    with open(BENCHMARK_JSON_PATH, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    if len(result_data) != len(benchmark_data):
        print(f"Error: Sample counts mismatch! Result count: {len(result_data)}, Benchmark count: {len(benchmark_data)}")
        sys.exit(1)

    for i, item in enumerate(result_data):
        result_times = item.get('result_time', [])
        if not isinstance(result_times, list):
            if result_times == -2:
                print(f"Error: Found unprocessed sample (result_time = -2) at index {i} (id: {item.get('id')}). Exiting.")
                sys.exit(1)
        else:
            if -2 in result_times:
                print(f"Error: Found unprocessed sample (result_time contains -2) at index {i} (id: {item.get('id')}). Exiting.")
                sys.exit(1)

    correct = 0
    total = len(result_data)
    
    for item in result_data:
        try:
            result_times = item.get('result_time', [-999, -999])
            if isinstance(result_times, list) and len(result_times) >= 2:
                r1 = float(result_times[0])
                r2 = float(result_times[1])
                
                answers = item.get('answer', [])
                if len(answers) >= 2:
                    t1 = float(answers[0]['trigger_time'])
                    t2 = float(answers[1]['trigger_time'])
                    
                    if abs(r1 - t1) <= 1.0 and abs(r2 - t2) <= 1.0:
                        correct += 1
        except Exception as e:
            pass

    print(f"Total evaluated samples: {total}")
    print(f"Correct samples: {correct}")
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Accuracy: N/A (Total is 0)")

if __name__ == "__main__":
    main()
