import json
import sys
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULT_FILE_PATH = os.path.join(PROJECT_ROOT, "Result", "Qwen3_VL_8B", "Proactive_Task_Management", "Task_Cancellation.json")
BENCHMARK_FILE_PATH = os.path.join(PROJECT_ROOT, "Benchmark", "Proactive_Task_Management", "Task_Cancellation.json")
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
        result_time = item.get('result_time')
        if result_time == -2:
            print(f"Error: Found unprocessed sample (result_time = -2) at index {i} (id: {item.get('id', 'Unknown')}). Exiting.")
            sys.exit(1)

    correct_time_only = 0
    correct_strict = 0
    total = len(result_data)
    
    for item in result_data:
        try:
            result_time = float(item.get('result_time', -999))
            trigger_time = float(item['answer'][0]['trigger_time'])
            
            time_match = abs(result_time - trigger_time) <= 1.0
            
            if time_match:
                correct_time_only += 1
                
                cancel_list = item.get('cancel', [])
                cancel_match = bool(cancel_list) and all(
                    'silence' in str(c).lower() and 'attention' not in str(c).lower() 
                    for c in cancel_list
                )
                
                if cancel_match:
                    correct_strict += 1
        except Exception as e:
            pass

    print(f"--- Evaluation Results ---")
    print(f"Total evaluated samples: {total}")
    print(f"\n[Condition 1] Result time is within trigger_time ± 1:")
    print(f"Correct samples: {correct_time_only}")
    if total > 0:
        accuracy_time = (correct_time_only / total) * 100
        print(f"Accuracy: {accuracy_time:.2f}%")
    else:
        print("Accuracy: N/A")
        
    print(f"\n[Condition 2] Time matches AND 'cancel' list contains only <silence>:")
    print(f"Correct samples: {correct_strict}")
    if total > 0:
        accuracy_strict = (correct_strict / total) * 100
        print(f"Accuracy: {accuracy_strict:.2f}%")
    else:
        print("Accuracy: N/A")

if __name__ == "__main__":
    main()
