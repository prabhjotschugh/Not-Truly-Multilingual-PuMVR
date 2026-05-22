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
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

class PuMVRKimiEvaluator:
    """
    Evaluator class for moonshotai/Kimi-VL-A3B-Instruct on PuMVR dataset.
    """
    SCRIPTS = ["gurmukhi", "shahmukhi", "roman"]
    SCRIPT_DISPLAY_NAMES = {
        "gurmukhi": "Gurmukhi (ਗੁਰਮੁਖੀ)",
        "shahmukhi": "Shahmukhi (شاہ مکھی)",
        "roman": "Roman (Punjabi)"
    }

    def __init__(
        self,
        model_id: str = "moonshotai/Kimi-VL-A3B-Instruct",
        batch_size: Optional[int] = 5,
        output_dir: Optional[str] = None,
        dataset_id: str = "Prabhjotschugh/PuMVR-Dataset",
        temperature: float = 0.2,
        max_tokens: int = 128,
        seed: int = 42,
        hf_token: str = HF_TOKEN
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        safe_model_name = model_id.replace(":", "_").replace("/", "_")
        self.output_dir = output_dir or f"./results_kimi_{safe_model_name}"
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
        """Log in to Hugging Face Hub"""
        if not self.hf_token or "xxxx" in self.hf_token:
            print(" WARNING: HF_TOKEN appears invalid. Please hardcode your token at the top of the script.")
        else:
            print(f" Authenticating with Hugging Face...")
            login(token=self.hf_token)

    def _set_seed(self):
        """Set random seed for reproducibility"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _create_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self):
        """Load Kimi-VL Model"""
        print(f" Loading Model: {self.model_id}")
        print(f"   Cache Dir: {os.environ['HF_HOME']}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
            if torch.cuda.is_available():
                print(f"   GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        except Exception as e:
            print(f"    Failed to load model: {e}")
            raise

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
        filename = f"res_kimi_{script}.csv"
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

    def _generate_answer(self, question: str, options: List[str], script: str, image: Image.Image) -> tuple[str, str]:
        """
        Generate answer using Kimi-VL logic.
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
            "Your answer:"
        )
        try:
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
                images=image,
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
        """Post-process extraction"""
        if raw_response == "[[ERROR]]" or raw_response.startswith("[[ERROR"):
            return raw_response
        cleaned = raw_response.strip()
        cleaned = cleaned.replace("**", "").replace("*", "").strip()
        for option in options:
            if cleaned.lower() == option.lower():
                return option
        for option in options:
            if option.lower() in cleaned.lower():
                return option
        for prefix in ["answer:", "Answer:", "your answer:", "Your Answer:"]:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        return cleaned

    def _process_script(self, script: str):
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
                    "question": row.get(f"scripts_{script}_question", ""),
                    "ground_truth": row.get(f"scripts_{script}_answer", ""),
                    "model_prediction": "[[ERROR]]",
                    "model_raw_response": f"[[ERROR: {str(e)}]]",
                    "is_correct": False,
                    "all_options": " | ".join(row.get(f"scripts_{script}_options", []))
                }
                self._save_single_result(error_entry, script)
            pbar.update(1)
        pbar.close()

    def _calculate_metrics(self) -> Dict[str, Any]:
        accuracies = {}
        counts = {}
        all_dfs = {}
        for script in self.SCRIPTS:
            filepath = self._get_csv_path(script)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                valid = df[~df['model_prediction'].astype(str).str.startswith("[[ERROR")]
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
        print(f"PuMVR Evaluator (Kimi-VL) | Model: {self.model_id}")
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
        print(f"Total Time        : {duration:.1f}s")
        print("="*60 + "\n")
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.summary, f, indent=2)
        print(f" Summary saved to {summary_path}")
        return self.summary

if __name__ == "__main__":
    model_name = "moonshotai/Kimi-VL-A3B-Instruct"
    evaluator = PuMVRKimiEvaluator(
        model_id=model_name,
        batch_size=None,
        output_dir="./results_exp1_kimi_vl_a3b"
    )
    evaluator.run()
