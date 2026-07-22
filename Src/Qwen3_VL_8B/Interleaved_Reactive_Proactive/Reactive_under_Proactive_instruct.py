import json
import os
import glob
import time
import torch
import threading
import sys
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

write_lock = threading.Lock()

def extract_attention(text: str) -> bool:
    if text is None:
        return False
    text_lower = text.lower()
    return "<attention>" in text_lower

def append_frames_to_payload(payload: list, sample_id: str, frame_list: list):
    for frame_idx in frame_list:
        frame_path = os.path.join(FRAME_BASE_DIR, sample_id, f"{frame_idx}.png")
        if not os.path.exists(frame_path):
            frame_path = os.path.join(FRAME_BASE_DIR, sample_id, f"{frame_idx}.jpg")
            
        if os.path.exists(frame_path):
            payload.append({
                "type": "image",
                "image": f"file://{frame_path}",
                "max_pixels": 360 * 420,
            })

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
        self.ans2 = ""
        self.attention_time = -1
        
        self.current_interval = 0  # 0: first question, 1: test attention 1, 2: second question (if valid), 3: test attention 2
        self.current_idx = 0
        self.done = False
        
        self.target_times_1 = []
        self.target_times_2 = []
        
        if self.valid:
            self.text1 = self.questions[0]['text']
            self.t1 = int(self.questions[0]['t'])
            
            self.text2 = self.questions[1]['text']
            self.t2 = int(self.questions[1]['t'])
            
            self.has_text3 = len(self.questions) > 2
            if self.has_text3:
                self.text3 = self.questions[2]['text']
                self.t3 = int(self.questions[2]['t'])
                
            
            if not self.has_text3:
                trigger_time = int(self.answers[0]['trigger_time'])
                start_test = max(int(self.answers[1]['trigger_time']), trigger_time - 4)
                end_test = min(self.video_ed, trigger_time + 4)
                if start_test <= end_test:
                    self.target_times_1 = list(range(start_test, end_test + 1))
                else:
                    self.target_times_1 = []
            else:
                self.target_times_1 = []
                
            if self.has_text3:
                trigger_time = int(self.answers[0]['trigger_time'])
                start_test = max(int(self.answers[2]['trigger_time']), trigger_time - 4)
                end_test = min(self.video_ed, trigger_time + 4)
                if start_test <= end_test:
                    self.target_times_2 = list(range(start_test, end_test + 1))
                else:
                    self.target_times_2 = []
        else:
            self.attention_time = -2
            self.done = True

def get_current_target_times(state: SampleState) -> List[int]:
    if state.current_interval == 1:
        return state.target_times_1
    elif state.current_interval == 3:
        return state.target_times_2
    return []

def prepare_message_for_state(state: SampleState, x: int) -> list:
    history = []
    
    if state.current_interval == 0:
        frame_start_1 = max(state.video_st, (state.t2 - 1) - MAX_FRAMES + 1)
        selected_frames_1 = list(range(frame_start_1, state.t2))
        
        content1 = [{"type": "text", "text": state.text1}]
        append_frames_to_payload(content1, state.sample_id, selected_frames_1)
        content1.append({"type": "text", "text": state.text2})
        
        history.append({"role": "user", "content": content1})
        
    elif state.current_interval == 1:
        frame_start_x = max(state.video_st, x - MAX_FRAMES + 1)
        selected_frames_x = list(range(frame_start_x, x + 1))
        
        frames_part1 = [f for f in selected_frames_x if f < state.t2]
        frames_part2 = [f for f in selected_frames_x if f >= state.t2]
        
        content1 = [{"type": "text", "text": state.text1}]
        append_frames_to_payload(content1, state.sample_id, frames_part1)
        content1.append({"type": "text", "text": state.text2})
        
        history.append({"role": "user", "content": content1})
        history.append({"role": "assistant", "content": [{"type": "text", "text": state.ans1}]})
        
        content2 = []
        append_frames_to_payload(content2, state.sample_id, frames_part2)
        content2.append({"type": "text", "text": "Now, considering all the instructions above, determine:\nAt the current moment, should you output <attention> or <silence>? You must output exactly: <attention> or <silence>. Do not output anything else."})
        
        history.append({"role": "user", "content": content2})
        
    elif state.current_interval == 2:
        frame_start_2 = max(state.video_st, (state.t3 - 1) - MAX_FRAMES + 1)
        selected_frames_2 = list(range(frame_start_2, state.t3))
        
        frames_part1 = [f for f in selected_frames_2 if f < state.t2]
        frames_part2 = [f for f in selected_frames_2 if f >= state.t2]
        
        content1 = [{"type": "text", "text": state.text1}]
        append_frames_to_payload(content1, state.sample_id, frames_part1)
        content1.append({"type": "text", "text": state.text2})
        
        history.append({"role": "user", "content": content1})
        history.append({"role": "assistant", "content": [{"type": "text", "text": state.ans1}]})
        
        content2 = []
        append_frames_to_payload(content2, state.sample_id, frames_part2)
        content2.append({"type": "text", "text": state.text3})
        
        history.append({"role": "user", "content": content2})
        
    elif state.current_interval == 3:
        frame_start_x = max(state.video_st, x - MAX_FRAMES + 1)
        selected_frames_x = list(range(frame_start_x, x + 1))
        
        frames_part1 = [f for f in selected_frames_x if f < state.t2]
        frames_part2 = [f for f in selected_frames_x if state.t2 <= f < state.t3]
        frames_part3 = [f for f in selected_frames_x if f >= state.t3]
        
        content1 = [{"type": "text", "text": state.text1}]
        append_frames_to_payload(content1, state.sample_id, frames_part1)
        content1.append({"type": "text", "text": state.text2})
        
        history.append({"role": "user", "content": content1})
        history.append({"role": "assistant", "content": [{"type": "text", "text": state.ans1}]})
        
        content2 = []
        append_frames_to_payload(content2, state.sample_id, frames_part2)
        content2.append({"type": "text", "text": "Now, considering all the instructions above, determine:\nAt the current moment, should you output <attention> or <silence>? You must output exactly: <attention> or <silence>. Do not output anything else."})
        content2.append({"type": "text", "text": state.text3})
        
        history.append({"role": "user", "content": content2})
        history.append({"role": "assistant", "content": [{"type": "text", "text": state.ans2}]})
        
        content3 = []
        append_frames_to_payload(content3, state.sample_id, frames_part3)
        content3.append({"type": "text", "text": "Now, considering all the instructions above, determine:\nAt the current moment, should you output <attention> or <silence>? You must output exactly: <attention> or <silence>. Do not output anything else."})
        
        history.append({"role": "user", "content": content3})
        
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
                    if 'result' in item and isinstance(item['result'], list) and len(item['result']) >= 3:
                        if item['result'][0] != -2:
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
                if state.current_interval in [0, 2]:
                    if state.current_interval == 2 and not state.has_text3:
                        state.current_interval += 1
                        continue
                    break
                else:
                    curr_times = get_current_target_times(state)
                    if not curr_times:
                        state.current_interval += 1
                        state.current_idx = 0
                        if state.current_interval > 3:
                            state.done = True
                    else:
                        break
                        
            if state.done:
                sample_copy = dict(state.sample)
                sample_copy['result'] = [state.attention_time, state.ans1, state.ans2]
                results[state.sample_id] = sample_copy
                
                sorted_data = [results[k] for k in sorted(results.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
                continue
                
            print(f"Starting sample {state.sample_id} Interval {state.current_interval}...")
            active_states.append(state)
            
        if not active_states:
            break
            
        batch_messages = []
        for state in active_states:
            if state.current_interval in [0, 2]:
                x = 0 # Not used in these intervals
            else:
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
        
        if image_inputs is None or len(image_inputs) == 0:
            image_inputs = None
        if video_inputs is None or len(video_inputs) == 0:
            video_inputs = None
            
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
            if state.current_interval in [0, 2]:
                x = 0
            else:
                curr_times = get_current_target_times(state)
                x = curr_times[state.current_idx]
            
            print(f"Sample {state.sample_id} Interval {state.current_interval} at X={x}: {out_text}")
            
            if state.current_interval == 0:
                state.ans1 = out_text
                if not state.has_text3:
                    state.current_interval = 1
                    state.current_idx = 0
                else:
                    state.current_interval = 2
                    state.current_idx = 0
            elif state.current_interval == 1:
                if extract_attention(out_text):
                    state.attention_time = x
                    state.done = True
                else:
                    state.current_idx += 1
                    if state.current_idx >= len(curr_times):
                        state.done = True
            elif state.current_interval == 2:
                state.ans2 = out_text
                state.current_interval = 3
                state.current_idx = 0
            elif state.current_interval == 3:
                if extract_attention(out_text):
                    state.attention_time = x
                    state.done = True
                else:
                    state.current_idx += 1
                    if state.current_idx >= len(curr_times):
                        state.done = True
                    
            while not state.done:
                if state.current_interval in [0, 2]:
                    if state.current_interval == 2 and not state.has_text3:
                        state.done = True
                    break
                else:
                    next_times = get_current_target_times(state)
                    if not next_times:
                        state.done = True
                    else:
                        break
                    
            if state.done:
                sample_copy = dict(state.sample)
                sample_copy['result'] = [state.attention_time, state.ans1, state.ans2]
                results[state.sample_id] = sample_copy
                
                sorted_data = [results[k] for k in sorted(results.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
                print(f"Completed sample {state.sample_id}, Result: {[state.attention_time, state.ans1, state.ans2]}")
            else:
                next_active_states.append(state)
                
        active_states = next_active_states

    print("All done!")

if __name__ == "__main__":
    main()
