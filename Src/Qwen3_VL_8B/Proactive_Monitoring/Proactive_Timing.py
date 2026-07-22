import json
import os
import glob
import time
import torch
from typing import List, Dict, Any
import argparse

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
TASK_GROUP = "Proactive_Monitoring"
TASK_NAME = "Proactive_Timing"
INPUT_JSON = os.path.join(PROJECT_ROOT, "Benchmark", TASK_GROUP, f"{TASK_NAME}.json")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "Result", "Qwen3_VL_8B", TASK_GROUP, "Proactive_Timing.json")
FRAME_BASE_DIR = os.path.join(PROJECT_ROOT, "Image", "1fps", TASK_GROUP, TASK_NAME)
MODEL_PATH = os.environ.get("QWEN3_VL_MODEL_PATH", "/mnt/bn/2d-videos/ljz/Streaming_AI/model/Qwen3-VL-8B-Instruct")
BATCH_SIZE = 16
MAX_TOKENS = 500
MAX_FRAMES = 16

def parse_args():
    parser = argparse.ArgumentParser(description=f"Run Qwen3-VL-8B on {TASK_GROUP}/{TASK_NAME}")
    parser.add_argument("--input_json", default=INPUT_JSON, help="Path to the benchmark JSON file")
    parser.add_argument("--output_json", default=OUTPUT_JSON, help="Path to the output JSON file")
    parser.add_argument("--frame_base_dir", default=FRAME_BASE_DIR, help="Path to the extracted frame directory")
    parser.add_argument("--model_path", default=MODEL_PATH, help="Path to the Qwen3-VL model")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for generation")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="Maximum new tokens")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES, help="Maximum frames per request")
    return parser.parse_args()

def extract_result(text: str) -> str:
    if text is None:
        return "Error: Empty response"
        
    text_lower = text.lower()
    if "<attention>" in text_lower:
        return "<attention>"
    elif "<silence>" in text_lower:
        return "<silence>"
    return text

class SampleState:
    def __init__(self, sample):
        self.sample = sample
        self.sample_id = str(sample['id'])
        self.video_st = int(sample['video_st'])
        self.video_ed = int(sample['video_ed'])
        trigger_time = int(sample['answer'][0]['trigger_time'])
        self.full_user_prompt = sample['question'][0]['text']
        
        start_time = max(self.video_st, trigger_time - 4)
        end_time = min(self.video_ed, trigger_time + 4)
        self.target_times = list(range(start_time, end_time + 1))
        self.current_idx = 0
        self.result_time = -1
        self.done = False

def main():
    global INPUT_JSON, OUTPUT_JSON, FRAME_BASE_DIR, MODEL_PATH, BATCH_SIZE, MAX_TOKENS, MAX_FRAMES
    args = parse_args()
    INPUT_JSON = args.input_json
    OUTPUT_JSON = args.output_json
    FRAME_BASE_DIR = args.frame_base_dir
    MODEL_PATH = args.model_path
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
                    processed_data[str(item['id'])] = item
        except Exception as e:
            print(f"Failed to load existing output json: {e}")

    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = 'left'
    print("Model loaded.")

    unprocessed_samples = [SampleState(s) for s in data if str(s['id']) not in processed_data]
    active_states = []

    results = processed_data.copy()

    print(f"Starting batch processing. Batch size: {BATCH_SIZE}, Samples to process: {len(unprocessed_samples)}")

    while unprocessed_samples or active_states:
        while len(active_states) < BATCH_SIZE and unprocessed_samples:
            state = unprocessed_samples.pop(0)
            print(f"Starting sample {state.sample_id}...")
            active_states.append(state)
            
        if not active_states:
            break
            
        batch_messages = []
        for state in active_states:
            x = state.target_times[state.current_idx]
            current_end = x
            
            all_frames = list(range(state.video_st, current_end + 1))
            if len(all_frames) > MAX_FRAMES:
                selected_frames = all_frames[-MAX_FRAMES:]
            else:
                selected_frames = all_frames
                
            video_frames = []
            for frame_idx in selected_frames:
                frame_path = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.jpg")
                if not os.path.exists(frame_path):
                    frame_path = os.path.join(FRAME_BASE_DIR, state.sample_id, f"{frame_idx}.png")
                
                if os.path.exists(frame_path):
                    video_frames.append(f"file://{frame_path}")
            
            content = []
            content.append({
                "type": "text",
                "text": state.full_user_prompt
            })
            
            if video_frames:
                for frame_path in video_frames:
                    content.append({
                        "type": "image",
                        "image": frame_path,
                        "max_pixels": 360 * 420,
                    })
            else:
                print(f"Warning: No frames collected for sample {state.sample_id} from {state.video_st} to {current_end}")
                
            message = {
                "role": "user",
                "content": content
            }
            batch_messages.append([message])
            
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
        for state, text in zip(active_states, output_texts):
            x = state.target_times[state.current_idx]
            parsed_result = extract_result(text)
            print(f"Sample {state.sample_id} at X={x}: {parsed_result}")
            
            if parsed_result == "<attention>":
                state.result_time = x
                state.done = True
            else:
                state.current_idx += 1
                if state.current_idx >= len(state.target_times):
                    state.result_time = -1
                    state.done = True
                    
            if state.done:
                sample_copy = dict(state.sample)
                sample_copy['result_time'] = state.result_time
                results[state.sample_id] = sample_copy
                
                sorted_data = [results[k] for k in sorted(results.keys(), key=lambda k: int(k))]
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(sorted_data, f, indent=4, ensure_ascii=False)
                print(f"Completed sample {state.sample_id}, Result Time: {state.result_time}")
            else:
                next_active_states.append(state)
                
        active_states = next_active_states

    print("All done!")

if __name__ == "__main__":
    main()
