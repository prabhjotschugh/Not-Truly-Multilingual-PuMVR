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
from PIL import Image
from datetime import datetime
from typing import Optional, Dict, List, Any, Set, Tuple
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

class PuMVRFewShotKimiEvaluator:
    SCRIPTS = ["gurmukhi", "shahmukhi", "roman"]
    SCRIPT_DISPLAY_NAMES = {
        "gurmukhi": "Gurmukhi (ਗੁਰਮੁਖੀ)",
        "shahmukhi": "Shahmukhi (شاہ مکھی)",
        "roman": "Roman (Punjabi)"
    }

    def __init__(
        self,
        model_id: str = "moonshotai/Kimi-VL-A3B-Instruct",
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
        self.output_dir = output_dir or f"./results_exp3_kimi_fewshot_{safe_model_name}"
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
        """Load Kimi-VL Model"""
        print(f" Loading Model: {self.model_id} (Few-Shot Transfer)")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=self.hf_token
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=self.hf_token
            )
            if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer.chat_template is None:
                self.processor.tokenizer.chat_template = self.tokenizer.chat_template
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                trust_remote_code=True,
                device_map="auto",
                token=self.hf_token
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
        filename = f"res_exp3_kimi_{safe_model_name}_{source_tag}_to_{target_script}.csv"
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

    def _construct_few_shot_prompt(self, shots: List[Dict], target_row: Dict, target_script: str) -> str:
        """
        Construct a multi-example prompt for Kimi-VL with few-shot examples.
        Note: Kimi doesn't have native multi-image support, so we use text description approach.
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
        few_shot_text = "I will show you some examples of the task. Then I will ask you a similar question.\n\n"
        for i, shot in enumerate(shots, 1):
            opt_str = "\n".join(shot['options'])
            few_shot_text += f"EXAMPLE {i}:\n"
            few_shot_text += f"Image: [Contains relevant visual information]\n"
            few_shot_text += f"Question: {shot['question']}\n"
            few_shot_text += f"Options:\n{opt_str}\n"
            few_shot_text += f"Correct Answer: {shot['answer']}\n\n"
        t_question = target_row[f"scripts_{target_script}_question"]
        t_options = target_row[f"scripts_{target_script}_options"]
        t_opt_str = "\n".join(t_options)
        target_text = f"NOW THE TARGET QUESTION:\n"
        target_text += f"Image: [Contains relevant visual information]\n"
        target_text += f"Question: {t_question}\n"
        target_text += f"Options:\n{t_opt_str}\n"
        target_text += f"System Instruction: {sys_prompt}\n"
        target_text += "Output ONLY the correct option text:"
        return few_shot_text + target_text

    def _generate_answer(self, shots: List[Dict], target_row: Dict, target_script: str) -> tuple[str, str]:
        """
        Generate answer using Kimi-VL with few-shot examples.
        Since Kimi doesn't support multiple images in one call, we use text description approach.
        """
        try:
            prompt_text = self._construct_few_shot_prompt(shots, target_row, target_script)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            text_input = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=text_input,
                images=target_row['image'],
                return_tensors="pt"
            ).to(self.model.device)
            generation_args = {
                "max_new_tokens": self.max_tokens,
                "do_sample": (self.temperature > 0),
                "temperature": self.temperature,
                "pad_token_id": self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
            }
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_args)
            input_length = inputs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[0][input_length:]
            response = self.processor.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)
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
    model_name = "moonshotai/Kimi-VL-A3B-Instruct"
    evaluator = PuMVRFewShotKimiEvaluator(
        model_id=model_name,
        k_shots=3,
        batch_size=None,
        output_dir="./results_exp3_kimi_fewshot_optimized"
    )
    evaluator.run_matrix_experiment()
    evaluator.run_mixed_experiment()
