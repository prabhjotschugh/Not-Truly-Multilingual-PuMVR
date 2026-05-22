import os
import sys
import time

CURRENT_DIR = "/"
CACHE_DIR = os.path.join(CURRENT_DIR, "hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
print(f" System: Caching datasets to {CACHE_DIR}")
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
import json
import pandas as pd
from PIL import Image
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, List, Any, Set
from tqdm import tqdm
from datasets import load_dataset
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class PuMVRGeminiEvaluator:
    """
    Evaluator class for Gemini models on PuMVR dataset.
    """
    SCRIPTS = ["gurmukhi", "shahmukhi", "roman"]
    SCRIPT_DISPLAY_NAMES = {
        "gurmukhi": "Gurmukhi (ਗੁਰਮੁਖੀ)",
        "shahmukhi": "Shahmukhi (شاہ مکھی)",
        "roman": "Roman (Punjabi)"
    }

    def __init__(
        self,
        model_id: str = "gemini-1.5-flash",
        batch_size: Optional[int] = 5,
        output_dir: Optional[str] = None,
        dataset_id: str = "Prabhjotschugh/PuMVR-Dataset",
        temperature: float = 0.1,
        seed: int = 42,
        api_key: str = GEMINI_API_KEY
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        safe_model_name = model_id.replace(":", "_").replace("/", "_")
        self.output_dir = output_dir or f"./results_gemini_{safe_model_name}"
        self.dataset_id = dataset_id
        self.temperature = temperature
        self.seed = seed
        self.api_key = api_key
        self._authenticate()
        self._create_output_dir()
        self._setup_gemini()
        self._load_dataset()
        self.summary: Dict[str, Any] = {}

    def _authenticate(self):
        """Configure Google Generative AI"""
        if not self.api_key or "AIza" not in self.api_key:
             print(" WARNING: GEMINI_API_KEY appears invalid. Please hardcode your key at the top.")
        else:
            print(f" Authenticating with Google Gemini API...")
            genai.configure(api_key=self.api_key)

    def _create_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _setup_gemini(self):
        """Setup the Gemini Model Object"""
        print(f" Configuring Model: {self.model_id}")
        self.generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            candidate_count=1,
            max_output_tokens=128
        )
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.model = genai.GenerativeModel(self.model_id)
        print(f"    Model configured.")

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
            print(f"    Target Processing: {self.process_limit} examples\n")
        except Exception as e:
            print(f"    Failed to load dataset: {e}\n")
            raise

    def _get_csv_path(self, script: str) -> str:
        filename = f"res_gemini_{script}.csv"
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

    def _generate_answer(self, question: str, options: List[str], script: str, image: Image.Image) -> tuple[str, str]:
        """
        Generate answer using Gemini API.
        Returns: (extracted_prediction, raw_response)
        """
        formatted_options = "\n".join(options)
        script_instruction = {
            "gurmukhi": "Answer in Gurmukhi script.",
            "shahmukhi": "Answer in Shahmukhi script.",
            "roman": "Answer in Roman script."
        }
        prompt_text = (
            f"Question: {question}\n"
            f"Options:\n{formatted_options}\n\n"
            f"Task: Identify the correct option based on the image.\n"
            f"Constraint: {script_instruction[script]} Copy the exact text of the correct option. Do not explain.\n"
            "CRITICAL RULES:\n"
            "1. Answer MUST be in the SAME script as the question\n"
            "2. Copy EXACTLY one option from above - character by character\n"
            "3. NO explanations, NO extra words, NO English translations\n"
            "4. NO letters like A), B), C) or numbers\n"
            "5. Output ONLY the option text, nothing else\n\n"
            "Your answer (copy exact text from options):"
            "Answer:"
        )
        try:
            inputs = [prompt_text, image]
            response = self.model.generate_content(
                inputs,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 return "[[BLOCKED]]", f"[[BLOCKED: {response.prompt_feedback.block_reason}]]"
            output_text = response.text
            return output_text.strip(), output_text
        except Exception as e:
            return "[[ERROR]]", f"[[ERROR: {str(e)}]]"

    def _extract_answer(self, raw_response: str, options: List[str]) -> str:
        """Post-process extraction"""
        if raw_response.startswith("[[ERROR") or raw_response.startswith("[[BLOCKED"):
            return raw_response
        cleaned = raw_response.strip()
        cleaned = cleaned.replace("**", "").replace("*", "")
        for option in options:
            if cleaned.lower() == option.lower():
                return option
        for option in options:
            if option.lower() in cleaned.lower():
                return option
        return cleaned

    def _process_script(self, script: str):
        """Process loop with Resume capability"""
        print(f"\n Processing script: {self.SCRIPT_DISPLAY_NAMES[script]}")
        completed_ids = self._load_completed_ids(script)
        if completed_ids:
            print(f"    Resuming: Found {len(completed_ids)} already completed.")
        dataset_subset = self.dataset.select(range(self.process_limit))
        pbar = tqdm(total=self.process_limit, desc=f"Evaluating {script}", unit="img")
        for row in dataset_subset:
            row_id = str(row['id'])
            if row_id in completed_ids:
                pbar.update(1)
                continue
            try:
                question = row[f"scripts_{script}_question"]
                options = row[f"scripts_{script}_options"]
                ground_truth = row[f"scripts_{script}_answer"]
                image = row['image']
                prediction, raw_response = self._generate_answer(question, options, script, image)
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
                    "is_correct": False
                }
                self._save_single_result(error_entry, script)
            pbar.update(1)
        pbar.close()

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculates metrics from the CSV files."""
        accuracies = {}
        counts = {}
        all_dfs = {}
        for script in self.SCRIPTS:
            filepath = self._get_csv_path(script)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                valid = df[~df['model_prediction'].astype(str).str.startswith("[[")]
                acc = valid['is_correct'].mean() if not valid.empty else 0.0
                accuracies[script] = acc
                counts[script] = len(df)
                all_dfs[script] = df
            else:
                accuracies[script] = 0.0
                counts[script] = 0
                all_dfs[script] = pd.DataFrame()
        scr = 0.0
        if not all_dfs[self.SCRIPTS[0]].empty:
            common_ids = set(all_dfs[self.SCRIPTS[0]]['id'])
            for script in self.SCRIPTS[1:]:
                if not all_dfs[script].empty:
                    common_ids = common_ids.intersection(set(all_dfs[script]['id']))
                else:
                    common_ids = set()
                    break
            if common_ids:
                consistent_count = 0
                for uid in common_ids:
                    if all(
                        all_dfs[s][all_dfs[s]['id'] == uid].iloc[0]['is_correct']
                        for s in self.SCRIPTS
                    ):
                        consistent_count += 1
                scr = consistent_count / len(common_ids)
        return {"accuracies": accuracies, "scr": scr, "counts": counts}

    def run(self) -> Dict[str, Any]:
        print("=" * 60)
        print(f"PuMVR Evaluator (Gemini API) | Model: {self.model_id}")
        print("=" * 60)
        start_time = datetime.now()
        for script in self.SCRIPTS:
            self._process_script(script)
        metrics = self._calculate_metrics()
        duration = (datetime.now() - start_time).total_seconds()
        self.summary = {
            "model": self.model_id,
            "metrics": metrics,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        for s in self.SCRIPTS:
            print(f"{s.capitalize():<15}: {metrics['accuracies'][s]:.2%}")
        print(f"Consistency (SCR): {metrics['scr']:.2%}")
        print(f"Total Time       : {duration:.1f}s")
        print("="*60 + "\n")
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.summary, f, indent=2)
        print(f" Summary saved to {summary_path}")
        return self.summary

if __name__ == "__main__":
    model_name = "gemini-2.5-flash"
    evaluator = PuMVRGeminiEvaluator(
        model_id=model_name,
        batch_size=375,
        output_dir="./results_gemini_exp1"
    )
    evaluator.run()
