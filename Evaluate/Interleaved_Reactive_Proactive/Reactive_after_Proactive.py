import json
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate R_after_P_Delay task")
    parser.add_argument("--result", default=os.path.join(PROJECT_ROOT, "Result", "Qwen3_VL_8B", "Interleaved_Reactive_Proactive", "Reactive_after_Proactive.json"), help="Path to the result JSON file")
    parser.add_argument("--benchmark", default=os.path.join(PROJECT_ROOT, "Benchmark", "Interleaved_Reactive_Proactive", "Reactive_after_Proactive.json"), help="Path to the benchmark JSON file")
    args = parser.parse_args()

    RESULT_JSON_PATH = args.result
    BENCHMARK_JSON_PATH = args.benchmark

    if not os.path.exists(RESULT_JSON_PATH):
        print(f"Error: Result file not found: {RESULT_JSON_PATH}")
        return
        
    if not os.path.exists(BENCHMARK_JSON_PATH):
        print(f"Error: Benchmark file not found: {BENCHMARK_JSON_PATH}")
        return

    with open(RESULT_JSON_PATH, "r", encoding="utf-8") as f:
        res_data = json.load(f)
        
    with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    for sample in res_data:
        res = sample.get("result", [])
        if res and len(res) > 1:
            if res[0] == -2 or res[1] == -2 or str(res[1]) == "-2":
                print(f"Error: Found sample with ID {sample.get('id')} having API error (-2) in result. Please re-run the inference script to fix errors.")
                return

    if len(res_data) != len(bench_data):
        print(f"Error: Sample count mismatch! Result has {len(res_data)} samples, Benchmark has {len(bench_data)} samples.")
        return

    bench_map = {item["id"]: item for item in bench_data}

    total_samples = len(res_data)
    reactive_correct_count = 0
    proactive_correct_count = 0
    fully_correct_count = 0

    for res_sample in res_data:
        s_id = res_sample["id"]
        if s_id not in bench_map:
            print(f"Warning: Result ID {s_id} not found in benchmark data.")
            continue
            
        bench_sample = bench_map[s_id]
        
        result_list = res_sample.get("result", [])
        if not result_list or len(result_list) < 2:
            continue
            
        predicted_time = result_list[0]
        predicted_text = str(result_list[1]).strip()
        
        bench_answers = bench_sample.get("answer", [])
        if not bench_answers or len(bench_answers) < 2:
            continue
            
        target_time = bench_answers[0].get("trigger_time", -1)
        target_text_list = bench_answers[1].get("text_list", [])
        
        is_reactive_correct = False
        pred_lower = predicted_text.lower()
        
        for tgt_text in target_text_list:
            tgt_lower = str(tgt_text).lower()
            if pred_lower in tgt_lower or tgt_lower in pred_lower:
                is_reactive_correct = True
                break
                
        if is_reactive_correct:
            reactive_correct_count += 1
            
        is_proactive_correct = False
        if target_time != -1 and (target_time - 1 <= predicted_time <= target_time + 1):
            is_proactive_correct = True
            proactive_correct_count += 1
            
        if is_reactive_correct and is_proactive_correct:
            fully_correct_count += 1

    reactive_accuracy = (reactive_correct_count / total_samples) * 100 if total_samples > 0 else 0
    proactive_accuracy = (proactive_correct_count / total_samples) * 100 if total_samples > 0 else 0
    fully_accuracy = (fully_correct_count / total_samples) * 100 if total_samples > 0 else 0

    print("=" * 50)
    print(f"R_after_P_Delay Evaluation Results:")
    print(f"Total Samples: {total_samples}")
    print("-" * 50)
    print(f"Proactive Correctness: {proactive_accuracy:.2f}% ({proactive_correct_count}/{total_samples})")
    print(f"Full Correctness:      {fully_accuracy:.2f}% ({fully_correct_count}/{total_samples})")
    print("=" * 50)

if __name__ == "__main__":
    evaluate()
