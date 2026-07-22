import json
import os
import glob
import time
import torch
from typing import List, Dict, Any
import argparse

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def parse_args():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    task_group = os.path.basename(os.path.dirname(__file__))
    task_name = os.path.splitext(os.path.basename(__file__))[0]
    benchmark_task_name = task_name[:-9] if task_name.endswith("_instruct") else task_name

    parser = argparse.ArgumentParser(description=f"Run Qwen3-VL-8B on {task_group}/{benchmark_task_name}")
    parser.add_argument(
        "--input_json",
        default=os.path.join(project_root, "Benchmark", task_group, f"{benchmark_task_name}.json"),
        help="Path to the benchmark JSON file",
    )
    parser.add_argument(
        "--output_json",
        default=os.path.join(project_root, "Result", "Qwen3_VL_8B", task_group, f"{task_name}.json"),
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--frame_base_dir",
        default=os.path.join(project_root, "Image", "1fps", task_group, benchmark_task_name),
        help="Path to the extracted frame directory",
    )
    parser.add_argument("--model_path", required=True, help="Path to the Qwen3-VL model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum new tokens")
    parser.add_argument("--max_frames", type=int, default=16, help="Maximum frames per request")
    return parser.parse_args()

def extract_attention(text: str) -> bool:
    if text is None:
        return False
    text_lower = text.lower()
    return "<attention>" in text_lower

class SampleState:
    def __init__(self, sample):
        self.sample = sample
        self.sample_id = str(sample['id'])
        self.video_st = int(sample.get('video_st', 0))
        self.video_ed = int(sample.get('video_ed', 0))
        
        self.questions = sample.get('question', [])
        self.answers = sample.get('answer', [])
        
        self.valid = len(self.questions) >= 2 and len(self.answers) >= 2
        
        self.ans1 = ""
        self.attention_time = -1
        
        self.current_interval = 0  # 0: first interval, 1: second interval
        self.current_idx = 0
        self.done = False
        
        self.target_times_1 = []
        self.target_times_2 = []
        
        if self.valid:
            self.text1 = self.questions[0]['text']
            self.t1 = int(self.questions[0]['t'])
            
            self.text2 = self.questions[1]['text']
            self.t2 = int(self.questions[1]['t'])
            
            trigger_time_2 = int(self.answers[1]['trigger_time'])
            
            self.target_times_1 = [self.t1 - 1]
            
            start_test = max(self.t2, trigger_time_2 - 4)
            end_test = min(self.video_ed, trigger_time_2 + 4)
            if start_test <= end_test:
                self.target_times_2 = list(range(start_test, end_test + 1))
        else:
            self.attention_time = -2
            self.done = True

def get_current_target_times(state: SampleState) -> List[int]:
    if state.current_interval == 0:
        return state.target_times_1
    else:
        return state.target_times_2

def prepare_message_for_state(state: SampleState, x: int) -> list:
    content = []
    history = []
    
    if state.current_interval == 0:
        frame_start_1 = max(state.video_st, (state.t1 - 1) - MAX_FRAMES + 1)
        selected_frames_1 = list(range(frame_start_1, state.t1))
        
        for frame_idx in selected_frames_1:
            frame_path_jpg = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.jpg")
            frame_path_png = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.png")
            frame_path = frame_path_jpg if os.path.exists(frame_path_jpg) else frame_path_png
            
            if os.path.exists(frame_path):
                content.append({
                    "type": "image",
                    "image": f"file://{frame_path}",
                    "max_pixels": 360 * 420,
                })
                
        content.append({
            "type": "text",
            "text": state.text1
        })
    elif state.current_interval == 1:
        frame_start_x = max(state.video_st, x - MAX_FRAMES + 1)
        selected_frames_x = list(range(frame_start_x, x + 1))
        
        frames_part1 = [f for f in selected_frames_x if f < state.t1]
        frames_part2 = [f for f in selected_frames_x if f == state.t1]
        frames_part3 = [f for f in selected_frames_x if f >= state.t2]
        
        for frame_idx in frames_part1:
            frame_path_jpg = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.jpg")
            frame_path_png = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.png")
            frame_path = frame_path_jpg if os.path.exists(frame_path_jpg) else frame_path_png
            
            if os.path.exists(frame_path):
                content.append({
                    "type": "image",
                    "image": f"file://{frame_path}",
                    "max_pixels": 360 * 420,
                })
                
        content.append({
            "type": "text",
            "text": state.text1
        })
        
        history.append({
            "role": "user",
            "content": content
        })
        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": state.ans1}]
        })
        content = []
        
        for frame_idx in frames_part2:
            frame_path_jpg = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.jpg")
            frame_path_png = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.png")
            frame_path = frame_path_jpg if os.path.exists(frame_path_jpg) else frame_path_png
            
            if os.path.exists(frame_path):
                content.append({
                    "type": "image",
                    "image": f"file://{frame_path}",
                    "max_pixels": 360 * 420,
                })
                
        content.append({
            "type": "text",
            "text": state.text2
        })
        
        for frame_idx in frames_part3:
            frame_path_jpg = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.jpg")
            frame_path_png = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.png")
            frame_path = frame_path_jpg if os.path.exists(frame_path_jpg) else frame_path_png
            
            if os.path.exists(frame_path):
                content.append({
                    "type": "image",
                    "image": f"file://{frame_path}",
                    "max_pixels": 360 * 420,
                })
                
    history.append({
        "role": "user",
        "content": content
    })
    return history

def main():
    global INPUT_JSON, OUTPUT_JSON, FRAME_BASE_DIR, BATCH_SIZE, MAX_TOKENS, MAX_FRAMES
    args = parse_args()
    INPUT_JSON = args.input_json
    OUTPUT_JSON = args.output_json
    FRAME_BASE_DIR = args.frame_base_dir
    BATCH_SIZE = args.batch_size
    MAX_TOKENS = args.max_tokens
    MAX_FRAMES = args.max_frames
    if not os.path.exists(INPUT_JSON):
        print(f"Input file not found: {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    processed_data = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for item in existing_data:
                    if 'result' in item and isinstance(item['result'], list) and len(item['result']) >= 2:
                        if item['result'][1] != -2:
                            processed_data[str(item['id'])] = item
        except Exception as e:
            print(f"Failed to load existing output json: {e}")

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = 'left'
    print("Model loaded.")

    unprocessed_samples = [SampleState(s) for s in data if str(s['id']) not in processed_data]
    
    active_states = []
    results = processed_data.copy()

    print(f"Starting batch processing. Batch size: {BATCH_SIZE}, Samples to process: {len(unprocessed_samples)}")

    while unprocessed_samples or active_states:
        while len(active_states) < BATCH_SIZE and unprocessed_samples:
            state = unprocessed_samples.pop(0)
            
            while not state.done:
                curr_times = get_current_target_times(state)
                if not curr_times:
                    state.current_interval += 1
                    state.current_idx = 0
                    if state.current_interval > 1:
                        state.done = True
                else:
                    break
                    
            if state.done:
                sample_copy = dict(state.sample)
                sample_copy['result'] = [state.ans1, state.attention_time]
                results[state.sample_id] = sample_copy
                
                sorted_data = [results[k] for k in sorted(results.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
                continue
                
            print(f"Starting sample {state.sample_id} Interval {state.current_interval + 1}...")
            active_states.append(state)
            
        if not active_states:
            break
            
        batch_messages = []
        for state in active_states:
            curr_times = get_current_target_times(state)
            x = curr_times[state.current_idx]
            
            history = prepare_message_for_state(state, x)
            batch_messages.append(history)
            
        text = processor.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        next_active_states = []
        for state, out_text in zip(active_states, output_texts):
            curr_times = get_current_target_times(state)
            x = curr_times[state.current_idx]
            
            print(f"Sample {state.sample_id} Interval {state.current_interval + 1} at X={x}: {out_text}")
            
            if state.current_interval == 0:
                state.ans1 = out_text
                state.current_interval += 1
                state.current_idx = 0
            else:
                if extract_attention(out_text):
                    state.attention_time = x
                    state.current_interval += 1
                    state.current_idx = 0
                else:
                    state.current_idx += 1
                    if state.current_idx >= len(curr_times):
                        state.current_interval += 1
                        state.current_idx = 0
                    
            while not state.done and state.current_interval <= 1:
                next_times = get_current_target_times(state)
                if not next_times:
                    state.current_interval += 1
                else:
                    break
                    
            if state.current_interval > 1:
                state.done = True
                    
            if state.done:
                sample_copy = dict(state.sample)
                sample_copy['result'] = [state.ans1, state.attention_time]
                results[state.sample_id] = sample_copy
                
                sorted_data = [results[k] for k in sorted(results.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
                print(f"Completed sample {state.sample_id}, Result: {[state.ans1, state.attention_time]}")
            else:
                next_active_states.append(state)
                
        active_states = next_active_states

    print("All done!")

if __name__ == "__main__":
    main()
