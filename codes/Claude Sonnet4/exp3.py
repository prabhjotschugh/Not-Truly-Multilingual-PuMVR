import os
import sys
import time
import json
import base64
import random
import pandas as pd
from PIL import Image
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, List, Any, Set
from tqdm import tqdm

CURRENT_DIR = os.getcwd()
CACHE_DIR = os.path.join(CURRENT_DIR, "hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
PROJECT_ID = "YOUR_PROJECT_ID_HERE"
REGION = "us-east5"
from datasets import load_dataset
from anthropic import AnthropicVertex

class PuMVRFewShotClaudeEvaluator:
    SCRIPTS = ["gurmukhi", "shahmukhi", "roman"]
    SCRIPT_DISPLAY_NAMES = {
        "gurmukhi": "Gurmukhi (ਗੁਰਮੁਖੀ)",
        "shahmukhi": "Shahmukhi (شاہ مکھی)",
        "roman": "Roman (Punjabi)"
    }

    def __init__(
        self,
        model_id: str = "claude-sonnet-4@20250514",
        k_shots: int = 3,
        batch_size: Optional[int] = 5,
        output_dir: Optional[str] = None,
        dataset_id: str = "Prabhjotschugh/PuMVR-Dataset",
        temperature: float = 0.1,
        max_tokens: int = 128,
        seed: int = 42,
        project_id: str = PROJECT_ID,
        region: str = REGION
    ):
        self.model_id = model_id
        self.k_shots = k_shots
        self.batch_size = batch_size
        safe_model_name = model_id.replace(":", "_").replace("/", "_").replace("@", "_")
        self.output_dir = output_dir or f"./results_exp3_claude_fewshot_{safe_model_name}"
        self.dataset_id = dataset_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.project_id = project_id
        self.region = region
        self._authenticate()
        self._set_seed()
        self._create_output_dir()
        self._load_dataset()
        self.results: Dict[str, pd.DataFrame] = {}
        self.summary: Dict[str, Any] = {}

    def _authenticate(self):
        """Configure Anthropic Client on Vertex AI"""
        print(f" Authenticating with Vertex AI...")
        print(f"   • Project: {self.project_id}")
        print(f"   • Region : {self.region}")
        try:
            self.client = AnthropicVertex(
                project_id=self.project_id,
                region=self.region
            )
        except Exception as e:
            print(f"    Authentication Failed: {e}")
            sys.exit(1)

    def _set_seed(self):
        """Set random seed for reproducibility (Python/Dataset sampling)"""
        random.seed(self.seed)

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
        """Get CSV file path"""
        safe_model_name = self.model_id.replace(":", "_").replace("/", "_").replace("@", "_")
        filename = f"res_exp3_claude_{safe_model_name}_{source_tag}_to_{target_script}.csv"
        return os.path.join(self.output_dir, filename)

    def _load_completed_ids(self, source_tag: str, target_script: str) -> Set[str]:
        """Reads the existing CSV to find which IDs are already done."""
        filepath = self._get_csv_path(source_tag, target_script)
        if not os.path.exists(filepath):
            return set()
        try:
            df = pd.read_csv(filepath)
            return set(df['id'].astype(str).tolist())
        except Exception:
            return set()

    def _save_single_result(self, result: Dict, source_tag: str, target_script: str):
        """Writes to CSV immediately after processing."""
        filepath = self._get_csv_path(source_tag, target_script)
        df_row = pd.DataFrame([result])
        header = not os.path.exists(filepath)
        try:
            df_row.to_csv(filepath, mode='a', header=header, index=False, encoding='utf-8')
        except Exception as e:
            print(f"\n    CRITICAL ERROR SAVING CSV: {e}")

    def _get_shots_uniform(self, target_id: str, script: str) -> List[Dict]:
        """Conditions 1 & 2: All shots from one specific script."""
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
        """Condition 3: Shots rotated through scripts."""
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

    def _process_image(self, image: Image.Image) -> str:
        """Resize and Encode Image to Base64."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        max_dim = 1500
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim))
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _construct_claude_inputs(self, shots: List[Dict], target_row: Dict, target_script: str):
        """
        Builds the System Prompt and Message List for Claude.
        """
        script_instruction = {
            "gurmukhi": "Gurmukhi (ਗੁਰਮੁਖੀ)",
            "shahmukhi": "Shahmukhi (شاہ مکھی)",
            "roman": "Roman Punjabi"
        }
        sys_prompt = (
            "You are a precise answering assistant. "
            "You will be given a visual question and options. "
            f"You must output ONLY the exact text of the correct option in {script_instruction[target_script]} script. "
            "CRITICAL RULES:\n"
            "1. NO reasoning.\n"
            "2. NO explanations.\n"
            "3. NO option letters (like A, B).\n"
            "4. Output strictly the option text."
        )
        messages = []
        for shot in shots:
            user_text = f"Question: {shot['question']}\nOptions:\n" + "\n".join(shot['options'])
            b64_img = self._process_image(shot['image'])
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_img,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_text
                    }
                ]
            })
            messages.append({
                "role": "assistant",
                "content": shot['answer']
            })
        t_q = target_row[f"scripts_{target_script}_question"]
        t_opt = "\n".join(target_row[f"scripts_{target_script}_options"])
        target_text = (
            f"Question: {t_q}\nOptions:\n{t_opt}\n\n"
            "Output ONLY the correct option text:"
        )
        b64_target_img = self._process_image(target_row['image'])
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64_target_img,
                    },
                },
                {
                    "type": "text",
                    "text": target_text
                }
            ]
        })
        return sys_prompt, messages

    def _generate_answer(self, sys_prompt: str, messages: List[Dict]) -> tuple[str, str]:
        """
        Generate answer using Claude API.
        """
        try:
            response = self.client.messages.create(
                model=self.model_id,
                system=sys_prompt,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            output_text = response.content[0].text
            return output_text.strip(), output_text
        except Exception as e:
            return "[[ERROR]]", f"[[ERROR: {str(e)}]]"

    def _extract_answer(self, raw_response: str, options: List[str]) -> str:
        """
        Post-process extraction (Identical to previous script)
        """
        if raw_response.startswith("[[ERROR"):
            return raw_response
        cleaned = raw_response.strip()
        prefixes = ["Answer:", "answer:", "The answer is:", "The correct answer is:"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
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
        """Evaluate a specific condition with resume capability"""
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
                sys_prompt, messages = self._construct_claude_inputs(shots, row, target_script)
                prediction, raw_response = self._generate_answer(sys_prompt, messages)
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
    GCP_PROJECT_ID = "YOUR_PROJECT_ID_HERE"
    GCP_REGION = "us-east5"
    MODEL_NAME = "claude-sonnet-4@20250514"
    evaluator = PuMVRFewShotClaudeEvaluator(
        model_id=MODEL_NAME,
        k_shots=3,
        batch_size=375,
        output_dir="./results_exp3_claude_fewshot",
        project_id=GCP_PROJECT_ID,
        region=GCP_REGION
    )
    evaluator.run_matrix_experiment()
    evaluator.run_mixed_experiment()
