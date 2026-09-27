"""
OpenJev Engine: Open System One Decision Scorer.
Based on daseinlabs/open-jev and razorback16/openjev.
Uses an open lightweight backbone (Qwen2.5-0.5B) for single-pass continuation logit scoring.
"""

import math
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class OpenJevScorer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading OpenJev backbone ({model_name}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map=self.device
        )
        self.model.eval()
        print("✓ OpenJev backbone ready.")

    def score_options(self, prompt: str, candidates: list[str]) -> list[float]:
        """
        Calculates log-likelihood of each candidate continuation given the prompt prefix.
        Returns softmax probabilities.
        """
        prefix_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        scores = []

        with torch.no_grad():
            for cand in candidates:
                cand_ids = self.tokenizer.encode(" " + cand.strip(), add_special_tokens=False)
                input_ids = torch.tensor([prefix_ids + cand_ids], device=self.device)
                
                outputs = self.model(input_ids)
                logits = outputs.logits[0] # (seq_len, vocab_size)

                # Log-prob of candidate tokens
                log_prob = 0.0
                cand_len = len(cand_ids)
                prefix_len = len(prefix_ids)
                for i in range(cand_len):
                    pos = prefix_len - 1 + i
                    token_id = cand_ids[i]
                    step_logits = logits[pos]
                    step_log_probs = torch.log_softmax(step_logits, dim=-1)
                    log_prob += step_log_probs[token_id].item()

                scores.append(log_prob / max(cand_len, 1))

        # Softmax over candidate normalized logprobs with temperature scaling
        scores_t = torch.tensor(scores)
        probs = torch.softmax(scores_t / 1.5, dim=-1).tolist()
        return probs

    def answer(self, state: str, questions: dict) -> dict:
        answers = {}
        for q_id, q in questions.items():
            q_type = q.get("type")
            instr = q.get("instructions", "")

            if q_type == "noul":
                prompt = f"Konteks: {state}\nPertanyaan: {instr}\nJawab Ya atau Tidak:\nJawaban:"
                probs = self.score_options(prompt, ["Tidak", "Ya"])
                # probs[1] = p(Ya / true)
                answers[q_id] = {
                    "type": "noul",
                    "noul": round(probs[1], 4),
                    "confidence": round(abs(probs[1] - 0.5) * 2, 4)
                }

            elif q_type == "choice":
                criteria = q.get("criteria", {})
                if isinstance(criteria, list):
                    labels = criteria
                    cand_descs = [str(c) for c in labels]
                elif isinstance(criteria, dict):
                    labels = list(criteria.keys())
                    cand_descs = [f"{k}: {criteria[k]}" for k in labels]
                else:
                    labels = q.get("options", ["opsi_1", "opsi_2"])
                    cand_descs = [str(c) for c in labels]

                prompt = f"Konteks: {state}\nPertanyaan: {instr}\nPilihlah kategori yang paling tepat:\n"
                for c in cand_descs:
                    prompt += f"- {c}\n"
                prompt += "Kategori terpilih:"

                probs = self.score_options(prompt, labels)
                best_idx = max(range(len(probs)), key=lambda i: probs[i])
                prob_map = {labels[i]: round(probs[i], 4) for i in range(len(labels))}
                
                # Confidence = (max_p - 1/K) / (1 - 1/K)
                K = len(labels)
                conf = 1.0 if K <= 1 else (max(probs) - 1/K) / (1 - 1/K)
                
                answers[q_id] = {
                    "type": "choice",
                    "choice": labels[best_idx],
                    "probabilities": prob_map,
                    "confidence": round(max(0.0, conf), 4)
                }

            elif q_type == "score":
                criteria = q.get("criteria", [])
                levels = [str(x) for x in criteria]
                prompt = f"Konteks: {state}\nPertanyaan: {instr}\nTingkat penilaian yang sesuai:\n"
                for idx, lvl in enumerate(levels):
                    prompt += f"{idx}. {lvl}\n"
                prompt += "Tingkat:"

                probs = self.score_options(prompt, [str(i) for i in range(len(levels))])
                exp_score = sum(i * p for i, p in enumerate(probs))
                answers[q_id] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    "legend": {str(i): levels[i] for i in range(len(levels))},
                    "probabilities": {str(i): round(probs[i], 4) for i in range(len(levels))},
                    "confidence": round(max(probs), 4)
                }

        return answers

if __name__ == "__main__":
    scorer = OpenJevScorer()
    res = scorer.answer("Paket rusak dan pecah", {
        "is_broken": {"type": "noul", "instructions": "Apakah barang rusak?"},
        "cat": {"type": "choice", "instructions": "Kategori?", "criteria": {"rusak": "Barang cacat", "lambat": "Keterlambatan"}}
    })
    import json
    print(json.dumps(res, indent=2))
