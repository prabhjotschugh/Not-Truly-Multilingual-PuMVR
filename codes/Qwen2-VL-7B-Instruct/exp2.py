import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(CURRENT_DIR, "hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
print(f" TACC Mode: Caching models to {CACHE_DIR}")
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
import json
import torch
import pandas as pd
from PIL import Image
from datetime import datetime
from typing import Optional, Dict, List, Any, Set
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class PuMVRBlindEvaluator:
    """
    Evaluator class for Experiment 2 (Blind Baseline) with Qwen2-VL.
    It functions identically to Experiment 1 but suppresses image input
    to test for text-only hallucinations.
    """
    SCRIPTS = ["gurmukhi", "shahmukhi", "roman"]
    SCRIPT_DISPLAY_NAMES = {
        "gurmukhi": "Gurmukhi (ਗੁਰਮੁਖੀ)",
        "shahmukhi": "Shahmukhi (شاہ مکھی)",
        "roman": "Roman (Punjabi)"
    }

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        batch_size: Optional[int] = 5,
        output_dir: Optional[str] = None,
        dataset_id: str = "Prabhjotschugh/PuMVR-Dataset",
        temperature: float = 0.1,
        max_tokens: int = 128,
        seed: int = 42,
        hf_token: str = HF_TOKEN
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        safe_model_name = model_id.replace(":", "_").replace("/", "_")
        self.output_dir = output_dir or f"./results_exp2_qwen_blind_{safe_model_name}"
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
        self.summary: Dict[str, Any] = {}

    def _authenticate(self):
        """Log in to Hugging Face Hub - SAME AS EXP1"""
        if not self.hf_token or "xxxx" in self.hf_token:
            print(" WARNING: HF_TOKEN appears invalid. Please hardcode your token at the top of the script.")
        else:
            print(f" Authenticating with Hugging Face...")
            login(token=self.hf_token)

    def _set_seed(self):
        """Set random seed for reproducibility - SAME AS EXP1"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _create_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self):
        """Load Qwen2-VL Model and Processor - SAME AS EXP1"""
        print(f" Loading Model: {self.model_id} (BLIND MODE - TEXT ONLY)")
        print(f"   Cache Dir: {os.environ['HF_HOME']}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                max_memory={0: "110GB"},
                token=self.hf_token,
                cache_dir=os.environ["HF_HOME"],
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                token=self.hf_token,
                cache_dir=os.environ["HF_HOME"],
                trust_remote_code=True
            )
            self.model.eval()
            print(f"    Model loaded on {self.model.device}")
            if torch.cuda.is_available():
                print(f"   GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"   GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        except Exception as e:
            print(f"    Failed to load model: {e}")
            raise

    def _load_dataset(self):
        """Load PuMVR dataset - SAME AS EXP1"""
        print(f" Loading dataset: {self.dataset_id}")
        try:
            self.dataset = load_dataset(
                self.dataset_id,
                split="train",
                cache_dir=os.environ["HF_DATASETS_CACHE"]
            )
            dataset_size = len(self.dataset)
            if self.batch_size is None:
                self.process_limit = dataset_size
            else:
                self.process_limit = min(self.batch_size, dataset_size)
            print(f"    Dataset loaded: {dataset_size} examples")
            print(f"    Target Processing: {self.process_limit} examples (Text Only)\n")
        except Exception as e:
            print(f"    Failed to load dataset: {e}\n")
            raise

    def _get_csv_path(self, script: str) -> str:
        """Get CSV file path - SAME AS EXP1 but with exp2 naming"""
        safe_model_name = self.model_id.replace(":", "_").replace("/", "_")
        filename = f"res_exp2_qwen_blind_{safe_model_name}_{script}.csv"
        return os.path.join(self.output_dir, filename)

    def _load_completed_ids(self, script: str) -> Set[str]:
        """Reads the existing CSV to find which IDs are already done. - SAME AS EXP1"""
        filepath = self._get_csv_path(script)
        if not os.path.exists(filepath):
            return set()
        try:
            df = pd.read_csv(filepath)
            return set(df['id'].astype(str).tolist())
        except Exception:
            return set()

    def _save_single_result(self, result: Dict, script: str):
        """
        CRITICAL: Writes to CSV immediately after processing. - SAME AS EXP1
        """
        filepath = self._get_csv_path(script)
        df_row = pd.DataFrame([result])
        header = not os.path.exists(filepath)
        try:
            df_row.to_csv(filepath, mode='a', header=header, index=False, encoding='utf-8')
        except Exception as e:
            print(f"\n    CRITICAL ERROR SAVING CSV: {e}")

    def _create_prompt(self, question: str, options: List[str], script: str) -> str:
        """
        Create prompt for blind evaluation - EXACTLY THE SAME AS YOUR PROMPT
        """
        formatted_options = "\n".join([opt for opt in options])
        script_instruction = {
            "gurmukhi": "ਗੁਰਮੁਖੀ ਵਿੱਚ",
            "shahmukhi": "شاہ مکھی وچ",
            "roman": "in Roman script"
        }
        prompt = (
            f"Question {script_instruction[script]}: {question}\n\n"
            f"Options:\n{formatted_options}\n\n"
            "CRITICAL RULES:\n"
            "1. Answer MUST be in the SAME script as the question\n"
            "2. Copy EXACTLY one option from above - character by character\n"
            "3. NO explanations, NO extra words, NO English translations\n"
            "4. NO letters like A), B), C) or numbers\n"
            "5. Output ONLY the option text, nothing else\n\n"
            "Your answer (copy exact text from options):"
        )
        return prompt

    def _generate_answer(self, question: str, options: List[str], script: str) -> tuple[str, str]:
        """
        Generate answer using Qwen2-VL logic - WITHOUT IMAGE INPUT
        Returns: (extracted_prediction, raw_response)
        """
        prompt_text = self._create_prompt(question, options, script)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        try:
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_prompt],
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=(self.temperature > 0)
                )
            generated_ids = [
                output_ids[len(inputs.input_ids[0]):] for output_ids in output_ids
            ]
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            return output_text.strip(), output_text
        except Exception as e:
            return "[[ERROR]]", f"[[ERROR: {str(e)}]]"

    def _extract_answer(self, raw_response: str, options: List[str]) -> str:
        """
        Post-process extraction - EXACTLY THE SAME AS YOUR CODE
        """
        if raw_response.startswith("[[ERROR"):
            return raw_response
        cleaned = raw_response.strip()
        prefixes = ["Answer:", "answer:", "The answer is:", "The correct answer is:"]
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        if len(cleaned) > 0 and cleaned[0].isalpha() and len(cleaned) > 1 and cleaned[1] in ['.', ')']:
            cleaned = cleaned[2:].strip()
        for option in options:
            if cleaned.lower() == option.lower():
                return option
        for option in options:
            if option.lower() in cleaned.lower() or cleaned.lower() in option.lower():
                return option
        return cleaned

    def _process_script(self, script: str):
        """Process loop with Resume capability - SIMILAR TO EXP1"""
        print(f"\n Processing script: {self.SCRIPT_DISPLAY_NAMES[script]} (Blind Mode)")
        completed_ids = self._load_completed_ids(script)
        if completed_ids:
            print(f"    Resuming: Found {len(completed_ids)} already completed.")
        dataset_subset = self.dataset.select(range(self.process_limit))
        pbar = tqdm(total=self.process_limit, desc=f"Evaluating {script} (Blind)", unit="img")
        for row in dataset_subset:
            row_id = str(row['id'])
            if row_id in completed_ids:
                pbar.update(1)
                continue
            try:
                question = row[f"scripts_{script}_question"]
                options = row[f"scripts_{script}_options"]
                ground_truth = row[f"scripts_{script}_answer"]
                prediction, raw_response = self._generate_answer(question, options, script)
                final_prediction = self._extract_answer(prediction, options)
                is_correct = (final_prediction == ground_truth)
                result_entry = {
                    "id": row['id'],
                    "category": row['category'],
                    "script": script,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_prediction": final_prediction,
                    "model_raw_response": raw_response,
                    "is_correct": is_correct,
                    "condition": "text_only_blind",
                    "all_options": " | ".join(options)
                }
                self._save_single_result(result_entry, script)
            except Exception as e:
                error_entry = {
                    "id": row['id'],
                    "category": row['category'],
                    "script": script,
                    "model_prediction": "[[ERROR]]",
                    "model_raw_response": f"[[ERROR: {str(e)}]]",
                    "is_correct": False,
                    "condition": "text_only_blind"
                }
                self._save_single_result(error_entry, script)
            pbar.update(1)
        pbar.close()

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculates metrics from the CSV files. - SIMILAR TO EXP1"""
        accuracies = {}
        counts = {}
        for script in self.SCRIPTS:
            filepath = self._get_csv_path(script)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                valid = df[~df['model_prediction'].astype(str).str.startswith("[[ERROR")]
                acc = valid['is_correct'].mean() if not valid.empty else 0.0
                accuracies[script] = acc
                counts[script] = len(df)
            else:
                accuracies[script] = 0.0
                counts[script] = 0
        return {"accuracies": accuracies, "counts": counts}

    def run(self) -> Dict[str, Any]:
        print("=" * 60)
        print(f"PuMVR Experiment 2: Blind Baseline (Qwen2-VL) | Model: {self.model_id}")
        print("=" * 60)
        start_time = datetime.now()
        for script in self.SCRIPTS:
            self._process_script(script)
        metrics = self._calculate_metrics()
        duration = (datetime.now() - start_time).total_seconds()
        self.summary = {
            "model": self.model_id,
            "experiment": "Exp2_Blind_Baseline",
            "metrics": metrics,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "temperature": self.temperature,
                "batch_size": self.batch_size,
                "max_tokens": self.max_tokens,
                "condition": "text_only_blind"
            }
        }
        print("\n" + "="*60)
        print("BLIND BASELINE SUMMARY")
        print("="*60)
        for s in self.SCRIPTS:
            print(f"{s.capitalize():<15}: {metrics['accuracies'][s]:.2%}")
        print(f"Total Time       : {duration:.1f}s")
        print("="*60 + "\n")
        summary_path = os.path.join(self.output_dir, "summary_exp2_blind.json")
        with open(summary_path, "w") as f:
            json.dump(self.summary, f, indent=2)
        print(f" Summary saved to {summary_path}")
        return self.summary

if __name__ == "__main__":
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    evaluator = PuMVRBlindEvaluator(
        model_id=model_name,
        batch_size=375,
        output_dir="./results_exp2_qwen_blind"
    )
    evaluator.run()
