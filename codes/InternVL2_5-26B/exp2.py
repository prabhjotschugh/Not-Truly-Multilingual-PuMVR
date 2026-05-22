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
from datetime import datetime
from typing import Optional, Dict, List, Any, Set
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel

class PuMVRBlindInternVLEvaluator:
    """
    Evaluator class for Experiment 2 (Blind Baseline) with InternVL2_5-26B.
    It functions identically to the standard evaluator but suppresses image input
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
        model_id: str = "OpenGVLab/InternVL2_5-26B",
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
        self.output_dir = output_dir or f"./results_exp3_internvl_blind_{safe_model_name}"
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
        if not self.hf_token or "xxxx" in self.hf_token:
            print(" WARNING: HF_TOKEN appears invalid.")
        else:
            print(f" Authenticating with Hugging Face...")
            login(token=self.hf_token)

    def _set_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _create_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self):
        """Load InternVL Model"""
        print(f" Loading Model: {self.model_id} (BLIND MODE - TEXT ONLY)")
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
        safe_model_name = self.model_id.replace(":", "_").replace("/", "_")
        filename = f"res_exp3_internvl_blind_{safe_model_name}_{script}.csv"
        return os.path.join(self.output_dir, filename)

    def _load_completed_ids(self, script: str) -> Set[str]:
        filepath = self._get_csv_path(script)
        if not os.path.exists(filepath):
            return set()
        try:
            df = pd.read_csv(filepath)
            return set(df['id'].astype(str).tolist())
        except Exception:
            return set()

    def _save_single_result(self, result: Dict, script: str):
        filepath = self._get_csv_path(script)
        df_row = pd.DataFrame([result])
        header = not os.path.exists(filepath)
        try:
            df_row.to_csv(filepath, mode='a', header=header, index=False, encoding='utf-8')
        except Exception as e:
            print(f"\n    CRITICAL ERROR SAVING CSV: {e}")

    def _create_prompt(self, question: str, options: List[str], script: str) -> str:
        """
        Constructs the BLIND prompt (Text-Only) exactly as requested.
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
        Generate answer using InternVL logic - WITHOUT IMAGE INPUT
        """
        prompt_text = self._create_prompt(question, options, script)
        try:
            generation_config = dict(
                max_new_tokens=self.max_tokens,
                do_sample=(self.temperature > 0),
                temperature=self.temperature,
            )
            with torch.no_grad():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values=None,
                    question=prompt_text,
                    generation_config=generation_config
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

    def _process_script(self, script: str):
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
        print(f"PuMVR Experiment 3: Blind Baseline (InternVL2.5) | Model: {self.model_id}")
        print("=" * 60)
        start_time = datetime.now()
        for script in self.SCRIPTS:
            self._process_script(script)
        metrics = self._calculate_metrics()
        duration = (datetime.now() - start_time).total_seconds()
        self.summary = {
            "model": self.model_id,
            "experiment": "Exp3_Blind_Baseline",
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
        print(f"Total Time        : {duration:.1f}s")
        print("="*60 + "\n")
        summary_path = os.path.join(self.output_dir, "summary_exp3_blind.json")
        with open(summary_path, "w") as f:
            json.dump(self.summary, f, indent=2)
        print(f" Summary saved to {summary_path}")
        return self.summary

if __name__ == "__main__":
    model_name = "OpenGVLab/InternVL2_5-26B"
    evaluator = PuMVRBlindInternVLEvaluator(
        model_id=model_name,
        batch_size=375,
        output_dir="./results_exp2_internvl_blind"
    )
    evaluator.run()
