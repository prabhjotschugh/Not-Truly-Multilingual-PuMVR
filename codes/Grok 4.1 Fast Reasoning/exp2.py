import os
import sys
import time

CURRENT_DIR = os.getcwd()
CACHE_DIR = os.path.join(CURRENT_DIR, "hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
print(f" System: Caching datasets to {CACHE_DIR}")
XAI_API_KEY = "YOUR_API_KEY_HERE"
import json
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Any, Set
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

class PuMVRBlindGrokEvaluator:
    """
    Evaluator class for Experiment 2 (Blind Baseline) with Grok API.
    Functions identically to Experiment 1 but suppresses image input
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
        model_id: str = "grok-4-1-fast-reasoning",
        batch_size: Optional[int] = 5,
        output_dir: Optional[str] = None,
        dataset_id: str = "Prabhjotschugh/PuMVR-Dataset",
        temperature: float = 0.1,
        seed: int = 42,
        api_key: str = XAI_API_KEY
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        safe_model_name = model_id.replace(":", "_").replace("/", "_")
        self.output_dir = output_dir or f"./results_exp2_grok_blind_{safe_model_name}"
        self.dataset_id = dataset_id
        self.temperature = temperature
        self.seed = seed
        self.api_key = api_key
        self.client = None
        self._authenticate()
        self._create_output_dir()
        self._load_dataset()
        self.summary: Dict[str, Any] = {}

    def _authenticate(self):
        """Configure xAI Client"""
        if not self.api_key or "YOUR_XAI_API_KEY" in self.api_key:
             print(" WARNING: XAI_API_KEY appears invalid. Please hardcode your key at the top.")
        else:
            print(f" Authenticating with xAI API...")
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1",
                )
                print(f"    Client initialized")
            except Exception as e:
                print(f"    Failed to initialize client: {e}")
                raise

    def _create_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_dataset(self):
        """Load PuMVR dataset"""
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
        """Get CSV file path"""
        safe_model_name = self.model_id.replace(":", "_").replace("/", "_")
        filename = f"res_exp2_grok_blind_{safe_model_name}_{script}.csv"
        return os.path.join(self.output_dir, filename)

    def _load_completed_ids(self, script: str) -> Set[str]:
        """Reads the existing CSV to find which IDs are already done."""
        filepath = self._get_csv_path(script)
        if not os.path.exists(filepath):
            return set()
        try:
            df = pd.read_csv(filepath)
            return set(df['id'].astype(str).tolist())
        except Exception:
            return set()

    def _save_single_result(self, result: Dict, script: str):
        """Writes to CSV immediately after processing."""
        filepath = self._get_csv_path(script)
        df_row = pd.DataFrame([result])
        header = not os.path.exists(filepath)
        try:
            df_row.to_csv(filepath, mode='a', header=header, index=False, encoding='utf-8')
        except Exception as e:
            print(f"\n    CRITICAL ERROR SAVING CSV: {e}")

    def _create_prompt(self, question: str, options: List[str], script: str) -> str:
        """Create prompt for blind evaluation"""
        formatted_options = "\n".join([opt for opt in options])
        script_instruction = {
            "gurmukhi": "ਗੁਰਮੁਖੀ ਵਿੱਚ",
            "shahmukhi": "شاہ مکھی وچ",
            "roman": "in Roman script"
        }
        prompt = (
            f"Question {script_instruction[script]}: {question}\n\n"
            f"Options:\n{formatted_options}\n\n"
            f"CRITICAL RULES:\n"
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
        Generate answer using Grok API - WITHOUT IMAGE INPUT
        Returns: (extracted_prediction, raw_response)
        """
        prompt_text = self._create_prompt(question, options, script)
        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=128
            )
            output_text = response.choices[0].message.content
            return output_text.strip(), output_text
        except Exception as e:
            return "[[ERROR]]", f"[[ERROR: {str(e)}]]"

    def _extract_answer(self, raw_response: str, options: List[str]) -> str:
        """Post-process extraction"""
        if raw_response.startswith("[[ERROR") or raw_response.startswith("[[BLOCKED"):
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
        """Process loop with Resume capability"""
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
        """Calculates metrics from the CSV files."""
        accuracies = {}
        counts = {}
        for script in self.SCRIPTS:
            filepath = self._get_csv_path(script)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                valid = df[~df['model_prediction'].astype(str).str.startswith("[[")]
                acc = valid['is_correct'].mean() if not valid.empty else 0.0
                accuracies[script] = acc
                counts[script] = len(df)
            else:
                accuracies[script] = 0.0
                counts[script] = 0
        return {"accuracies": accuracies, "counts": counts}

    def run(self) -> Dict[str, Any]:
        print("=" * 60)
        print(f"PuMVR Experiment 2: Blind Baseline (Grok API) | Model: {self.model_id}")
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
    model_name = "grok-4-1-fast-reasoning"
    evaluator = PuMVRBlindGrokEvaluator(
        model_id=model_name,
        batch_size=5,
        output_dir="./results_exp2_grok_blind"
    )
    evaluator.run()
