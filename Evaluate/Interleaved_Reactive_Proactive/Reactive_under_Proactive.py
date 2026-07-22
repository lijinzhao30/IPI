import json
import os
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate R_under_P task")
    parser.add_argument("--result", default=os.path.join(PROJECT_ROOT, "Result", "Qwen3_VL_8B", "Interleaved_Reactive_Proactive", "Reactive_under_Proactive.json"), help="Path to the result JSON file")
    parser.add_argument("--benchmark", default=os.path.join(PROJECT_ROOT, "Benchmark", "Interleaved_Reactive_Proactive", "Reactive_under_Proactive.json"), help="Path to the benchmark JSON file")
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
        if res and len(res) > 0 and res[0] == -2:
            print(f"Error: Found sample with ID {sample.get('id')} having result[0] == -2. Please re-run the inference script to fix API errors.")
            return

    if len(res_data) != len(bench_data):
        print(f"Error: Sample count mismatch! Result has {len(res_data)} samples, Benchmark has {len(bench_data)} samples.")
        return

    bench_map = {item["id"]: item for item in bench_data}

    total_samples = len(res_data)
    time_correct_count = 0
    strict_correct_score = 0.0

    for res_sample in res_data:
        s_id = res_sample["id"]
        if s_id not in bench_map:
            print(f"Warning: Result ID {s_id} not found in benchmark data.")
            continue
            
        bench_sample = bench_map[s_id]
        
        result_list = res_sample.get("result", [])
        if not result_list or len(result_list) == 0:
            continue
            
        predicted_time = result_list[0]
        
        bench_answers = bench_sample.get("answer", [])
        if not bench_answers or len(bench_answers) == 0:
            continue
            
        target_time = bench_answers[0].get("trigger_time", -1)
        
        is_time_correct = False
        if target_time != -1 and (target_time - 1 <= predicted_time <= target_time + 1):
            is_time_correct = True
            time_correct_count += 1
            
        if is_time_correct:
            
            def check_text_match(pred_text, target_list):
                if not pred_text or not target_list:
                    return False
                pred_lower = str(pred_text).lower()
                for target_text in target_list:
                    tgt_lower = str(target_text).lower()
                    if pred_lower in tgt_lower or tgt_lower in pred_lower:
                        return True
                return False

            ans1_correct = False
            if len(result_list) > 1 and len(bench_answers) > 1:
                pred1 = result_list[1]
                target_list1 = bench_answers[1].get("text_list", [])
                ans1_correct = check_text_match(pred1, target_list1)
                
            has_ans2 = len(result_list) > 2 and len(bench_answers) > 2
            ans2_correct = False
            if has_ans2:
                pred2 = result_list[2]
                target_list2 = bench_answers[2].get("text_list", [])
                ans2_correct = check_text_match(pred2, target_list2)
                
            if has_ans2:
                if ans1_correct and ans2_correct:
                    strict_correct_score += 1.0
                elif ans1_correct or ans2_correct:
                    strict_correct_score += 0.5
            else:
                if ans1_correct:
                    strict_correct_score += 1.0

    time_accuracy = (time_correct_count / total_samples) * 100 if total_samples > 0 else 0
    strict_accuracy = (strict_correct_score / total_samples) * 100 if total_samples > 0 else 0

    print("=" * 40)
    print(f"Evaluation Results:")
    print(f"Total Samples: {total_samples}")
    print("-" * 40)
    print(f"Time Correctness:   {time_accuracy:.2f}% ({time_correct_count}/{total_samples})")
    print(f"Strict Correctness: {strict_accuracy:.2f}% (Score: {strict_correct_score}/{total_samples})")
    print("=" * 40)

if __name__ == "__main__":
    evaluate()
