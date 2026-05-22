import os
import sys
import time
import base64
import json
import pandas as pd
from PIL import Image
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, List, Any, Set
from tqdm import tqdm
from datasets import load_dataset
from anthropic import AnthropicVertex

CURRENT_DIR = os.getcwd()
CACHE_DIR = os.path.join(CURRENT_DIR, "hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
print(f" System: Caching datasets to {CACHE_DIR}")
PROJECT_ID = "YOUR_PROJECT_ID_HERE"
LOCATION = "us-central1"

class PuMVRClaudeEvaluator:
    """
    Evaluator class for Claude models on Vertex AI for PuMVR dataset.
    """
    SCRIPTS = ["gurmukhi", "shahmukhi", "roman"]
    SCRIPT_DISPLAY_NAMES = {
        "gurmukhi": "Gurmukhi (ਗੁਰਮੁਖੀ)",
        "shahmukhi": "Shahmukhi (شاہ مکھی)",
        "roman": "Roman (Punjabi)"
    }

    def __init__(
        self,
        model_id: str = "claude-3-5-sonnet-v2@20241022",
        batch_size: Optional[int] = 5,
        output_dir: Optional[str] = None,
        dataset_id: str = "Prabhjotschugh/PuMVR-Dataset",
        temperature: float = 0.1,
        project_id: str = PROJECT_ID,
        location: str = LOCATION
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        safe_model_name = model_id.replace(":", "_").replace("/", "_").replace("@", "_")
        self.output_dir = output_dir or f"./results_claude_{safe_model_name}"
        self.dataset_id = dataset_id
        self.temperature = temperature
        self.project_id = project_id
        self.location = location
        self._authenticate()
        self._create_output_dir()
        self._load_dataset()
        self.summary: Dict[str, Any] = {}

    def _authenticate(self):
        """Configure Anthropic Client on Vertex AI"""
        print(f" Authenticating with Vertex AI (Project: {self.project_id}, Loc: {self.location})...")
        try:
            self.client = AnthropicVertex(
                region=self.location,
                project_id=self.project_id
            )
            print("    Client initialized.")
        except Exception as e:
            print(f"    Authentication Failed: {e}")
            print("      Ensure you ran 'gcloud auth application-default login'")
            sys.exit(1)

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
            print(f"    Target Processing: {self.process_limit} examples\n")
        except Exception as e:
            print(f"    Failed to load dataset: {e}\n")
            raise

    def _get_csv_path(self, script: str) -> str:
        filename = f"res_claude_{script}.csv"
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

    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to Base64 string"""
        buffered = BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _generate_answer(self, question: str, options: List[str], script: str, image: Image.Image) -> tuple[str, str]:
        """
        Generate answer using Claude on Vertex AI.
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
            b64_image = self._encode_image(image)
            message = self.client.messages.create(
                model=self.model_id,
                max_tokens=128,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt_text
                            }
                        ],
                    }
                ],
            )
            output_text = message.content[0].text
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
        print(f"PuMVR Evaluator (Vertex AI / Claude) | Model: {self.model_id}")
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
    model_name = "claude-sonnet-4@20250514"
    evaluator = PuMVRClaudeEvaluator(
        model_id=model_name,
        batch_size=None,
        output_dir="./results_claude_exp1",
        project_id="YOUR_PROJECT_ID_HERE",
        location="YOUR_LOCATION_HERE"
    )
    evaluator.run()
