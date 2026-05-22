import os
import sys
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "YOUR_ENV_VAR_HERE"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(CURRENT_DIR, "hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
print(f" TACC Mode: Caching models to {CACHE_DIR}")
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
import json
import torch
import random
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from datetime import datetime
from typing import Optional, Dict, List, Any, Set, Tuple
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel

def build_transform(input_size):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(list(target_ratios), key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height), Image.LANCZOS)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size), Image.LANCZOS)
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class PuMVRFewShotInternVLEvaluator:
    SCRIPTS = ["gurmukhi", "shahmukhi", "roman"]
    SCRIPT_DISPLAY_NAMES = {
        "gurmukhi": "Gurmukhi (ਗੁਰਮੁਖੀ)",
        "shahmukhi": "Shahmukhi (شاہ مکھی)",
        "roman": "Roman (Punjabi)"
    }

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2_5-26B",
        k_shots: int = 3,
        batch_size: Optional[int] = 5,
        output_dir: Optional[str] = None,
        dataset_id: str = "Prabhjotschugh/PuMVR-Dataset",
        temperature: float = 0.1,
        max_tokens: int = 128,
        seed: int = 42,
        hf_token: str = HF_TOKEN
    ):
        self.model_id = model_id
        self.k_shots = k_shots
        self.batch_size = batch_size
        safe_model_name = model_id.replace(":", "_").replace("/", "_")
        self.output_dir = output_dir or f"./results_exp3_internvl_fewshot_{safe_model_name}"
        self.dataset_id = dataset_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.hf_token = hf_token
        self._authenticate()
        self._set_seed()
        self._create_output_dir()
        self._load_model()
        self._load_dataset()
        self.results: Dict[str, pd.DataFrame] = {}
        self.summary: Dict[str, Any] = {}

    def _authenticate(self):
        if not self.hf_token or "xxxx" in self.hf_token:
            print(" WARNING: HF_TOKEN appears invalid.")
        else:
            print(f" Authenticating with Hugging Face...")
            login(token=self.hf_token)

    def _set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _create_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self):
        """Load InternVL Model"""
        print(f" Loading Model: {self.model_id} (Few-Shot Transfer)")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=False,
                token=self.hf_token
            )
            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                token=self.hf_token,
                device_map="cuda"
            ).eval()
            print(f"    Model loaded on {self.model.device}")
        except Exception as e:
            print(f"    Failed to load model: {e}")
            raise

    def _load_dataset(self):
        print(f" Loading dataset: {self.dataset_id}")
        try:
            self.dataset = load_dataset(
                self.dataset_id,
                split="train",
                cache_dir=os.environ["HF_DATASETS_CACHE"]
            )
            self.dataset_list = [item for item in self.dataset]
            if self.batch_size is None:
                self.process_limit = len(self.dataset)
            else:
                self.process_limit = min(self.batch_size, len(self.dataset))
            print(f"    Dataset loaded: {len(self.dataset)} examples")
            print(f"    Target Processing: {self.process_limit} examples (Few-Shot)")
            print(f"    Few-Shot Count (k): {self.k_shots}\n")
        except Exception as e:
            print(f"    Failed to load dataset: {e}\n")
            raise

    def _get_csv_path(self, source_tag: str, target_script: str) -> str:
        safe_model_name = self.model_id.replace(":", "_").replace("/", "_")
        filename = f"res_exp3_internvl_{safe_model_name}_{source_tag}_to_{target_script}.csv"
        return os.path.join(self.output_dir, filename)

    def _load_completed_ids(self, source_tag: str, target_script: str) -> Set[str]:
        filepath = self._get_csv_path(source_tag, target_script)
        if not os.path.exists(filepath):
            return set()
        try:
            df = pd.read_csv(filepath)
            return set(df['id'].astype(str).tolist())
        except Exception:
            return set()

    def _save_single_result(self, result: Dict, source_tag: str, target_script: str):
        filepath = self._get_csv_path(source_tag, target_script)
        df_row = pd.DataFrame([result])
        header = not os.path.exists(filepath)
        try:
            df_row.to_csv(filepath, mode='a', header=header, index=False, encoding='utf-8')
        except Exception as e:
            print(f"\n    CRITICAL ERROR SAVING CSV: {e}")

    def _get_shots_uniform(self, target_id: str, script: str) -> List[Dict]:
        candidates = [x for x in self.dataset_list if x['id'] != target_id]
        shots = random.sample(candidates, self.k_shots)
        formatted_shots = []
        for shot in shots:
            formatted_shots.append({
                "image": shot['image'],
                "question": shot[f"scripts_{script}_question"],
                "options": shot[f"scripts_{script}_options"],
                "answer": shot[f"scripts_{script}_answer"],
                "script_tag": script
            })
        return formatted_shots

    def _get_shots_mixed(self, target_id: str) -> List[Dict]:
        candidates = [x for x in self.dataset_list if x['id'] != target_id]
        raw_shots = random.sample(candidates, self.k_shots)
        formatted_shots = []
        script_cycle = self.SCRIPTS * (self.k_shots // len(self.SCRIPTS) + 1)
        for i, shot in enumerate(raw_shots):
            assigned_script = script_cycle[i]
            formatted_shots.append({
                "image": shot['image'],
                "question": shot[f"scripts_{assigned_script}_question"],
                "options": shot[f"scripts_{assigned_script}_options"],
                "answer": shot[f"scripts_{assigned_script}_answer"],
                "script_tag": assigned_script
            })
        return formatted_shots

    def _construct_text_prompt(self, q: str, opts: List[str], sys_prompt: str) -> str:
        """Helper to format a single turn's text component"""
        opt_str = "\n".join(opts)
        full_text = f"<image>\n{sys_prompt}\nQuestion: {q}\nOptions:\n{opt_str}\n"
        return full_text

    def _generate_answer(self, shots: List[Dict], target_row: Dict, target_script: str) -> tuple[str, str]:
        """
        Generate answer using InternVL with multi-image history (Few-Shot).
        OPTIMIZED: Uses max_num=4 for shots and max_num=8 for target to save memory.
        """
        script_instruction = {
            "gurmukhi": "Gurmukhi (ਗੁਰਮੁਖੀ)",
            "shahmukhi": "Shahmukhi (شاہ مکھی)",
            "roman": "Roman Punjabi"
        }
        sys_prompt = (
            f"Constraint: Output ONLY the exact text of the correct option in {script_instruction[target_script]} script.\n"
            "CRITICAL RULES:\n"
            "1. NO reasoning.\n"
            "2. NO explanations.\n"
            "3. NO option letters (like A, B).\n"
            "4. Output strictly the option text."
        )
        try:
            pixel_list = []
            for shot in shots:
                px = load_image(shot['image'], max_num=4).to(torch.bfloat16).cuda()
                pixel_list.append(px)
            target_px = load_image(target_row['image'], max_num=8).to(torch.bfloat16).cuda()
            pixel_list.append(target_px)
            pixel_values = torch.cat(pixel_list, dim=0)
            history = []
            for shot in shots:
                shot_q_text = self._construct_text_prompt(
                    shot['question'],
                    shot['options'],
                    "Task: Select the correct option."
                )
                shot_a_text = shot['answer']
                history.append((shot_q_text, shot_a_text))
            t_q = target_row[f"scripts_{target_script}_question"]
            t_opt = target_row[f"scripts_{target_script}_options"]
            target_question = self._construct_text_prompt(t_q, t_opt, sys_prompt)
            target_question += "\nOutput ONLY the correct option text:"
            generation_config = dict(
                max_new_tokens=self.max_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature,
            )
            with torch.no_grad():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    target_question,
                    generation_config,
                    history=history
                )
            return response.strip(), response
        except Exception as e:
            return "[[ERROR]]", f"[[ERROR: {str(e)}]]"

    def _extract_answer(self, raw_response: str, options: List[str]) -> str:
        if raw_response.startswith("[[ERROR"):
            return raw_response
        cleaned = raw_response.strip()
        prefixes = ["Answer:", "answer:", "The answer is:", "The correct answer is:"]
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        if len(cleaned) > 0 and cleaned[0].isalpha() and len(cleaned) > 1 and cleaned[1] in ['.', ')']:
            cleaned = cleaned[2:].strip()
        cleaned = cleaned.replace("**", "").replace("*", "")
        for option in options:
            if cleaned.lower() == option.lower():
                return option
        for option in options:
            if option.lower() in cleaned.lower() or cleaned.lower() in option.lower():
                return option
        return cleaned

    def _evaluate_condition(self, source_tag: str, target_script: str):
        print(f"\n Running: Source[{source_tag.upper()}] -> Target[{target_script.upper()}]")
        completed_ids = self._load_completed_ids(source_tag, target_script)
        if completed_ids:
            print(f"    Resuming: Found {len(completed_ids)} already completed.")
        dataset_subset = self.dataset.select(range(self.process_limit))
        correct_count = 0
        total_processed = 0
        pbar = tqdm(total=self.process_limit, desc=f"Evaluating {source_tag}→{target_script}", unit="img")
        for row in dataset_subset:
            row_id = str(row['id'])
            if row_id in completed_ids:
                pbar.update(1)
                continue
            try:
                if source_tag == "mixed":
                    shots = self._get_shots_mixed(row['id'])
                else:
                    shots = self._get_shots_uniform(row['id'], source_tag)
                prediction, raw_response = self._generate_answer(shots, row, target_script)
                ground_truth = row[f"scripts_{target_script}_answer"].strip()
                final_prediction = self._extract_answer(prediction, row[f"scripts_{target_script}_options"])
                is_correct = (final_prediction == ground_truth)
                if not is_correct:
                    if ground_truth in final_prediction:
                        is_correct = True
                if is_correct: correct_count += 1
                total_processed += 1
                result_entry = {
                    "id": row['id'],
                    "category": row['category'],
                    "source_mode": source_tag,
                    "target_script": target_script,
                    "ground_truth": ground_truth,
                    "model_prediction": final_prediction,
                    "model_raw_response": raw_response,
                    "is_correct": is_correct,
                    "k_shots": self.k_shots,
                    "condition": "few_shot_transfer",
                    "all_options": " | ".join(row[f"scripts_{target_script}_options"])
                }
                self._save_single_result(result_entry, source_tag, target_script)
                if not raw_response.startswith("[[ERROR"):
                    status = "" if is_correct else "✗"
                    pbar.set_postfix({"Status": status, "ID": row['id']})
                else:
                    pbar.set_postfix({"Status": "", "ID": row['id']})
            except Exception as e:
                error_entry = {
                    "id": row['id'],
                    "category": row['category'],
                    "source_mode": source_tag,
                    "target_script": target_script,
                    "model_prediction": "[[ERROR]]",
                    "model_raw_response": f"[[ERROR: {str(e)}]]",
                    "is_correct": False,
                    "k_shots": self.k_shots,
                    "condition": "few_shot_transfer"
                }
                self._save_single_result(error_entry, source_tag, target_script)
                pbar.set_postfix({"Status": "", "ID": row['id']})
            if 'prediction' in locals(): del prediction
            if 'raw_response' in locals(): del raw_response
            if 'shots' in locals(): del shots
            torch.cuda.empty_cache()
            gc.collect()
            pbar.update(1)
        pbar.close()
        accuracy = (correct_count / total_processed) if total_processed > 0 else 0.0
        print(f"   Accuracy: {accuracy:.1%} ({correct_count}/{total_processed})")
        return accuracy

    def run_matrix_experiment(self):
        """Conditions 1 & 2: 3x3 Matrix"""
        print("\n" + "="*60)
        print("STARTING MATRIX EXPERIMENT (Conditions 1 & 2)")
        print("="*60)
        matrix_scores = {}
        for source in self.SCRIPTS:
            matrix_scores[source] = {}
            for target in self.SCRIPTS:
                acc = self._evaluate_condition(source, target)
                matrix_scores[source][target] = acc
        summary_path = os.path.join(self.output_dir, "matrix_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "model": self.model_id,
                "k_shots": self.k_shots,
                "temperature": self.temperature,
                "matrix_scores": matrix_scores,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        print(f"\n Matrix summary saved to {summary_path}")
        return matrix_scores

    def run_mixed_experiment(self):
        """Condition 3: Mixed Source"""
        print("\n" + "="*60)
        print("STARTING MIXED EXPERIMENT (Condition 3)")
        print("="*60)
        mixed_scores = {}
        for target in self.SCRIPTS:
            acc = self._evaluate_condition("mixed", target)
            mixed_scores[target] = acc
        summary_path = os.path.join(self.output_dir, "mixed_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "model": self.model_id,
                "k_shots": self.k_shots,
                "temperature": self.temperature,
                "mixed_scores": mixed_scores,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        print(f"\n Mixed summary saved to {summary_path}")
        return mixed_scores

if __name__ == "__main__":
    model_name = "OpenGVLab/InternVL2_5-26B"
    evaluator = PuMVRFewShotInternVLEvaluator(
        model_id=model_name,
        k_shots=3,
        batch_size=None,
        output_dir="./results_exp3_internvl_fewshot_optimized"
    )
    evaluator.run_matrix_experiment()
    evaluator.run_mixed_experiment()
