import json
import csv
import random
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    # optional Llama/transformers integration
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False


class MeditationRecommender:
    """Produces a structured meditation plan from a fused decision.

    Behavior:
    - Reads fused decision JSON to obtain recommended meditation type and confidence.
    - Loads meditation catalog (`meditation.csv`) for base templates.
    - If transformers/LLama is available and enabled, calls a chat/generation pipeline to produce
      tailored scripts and steps. Otherwise falls back to CSV templates + simple heuristics.
    """

    def __init__(self, use_llama: bool = False, llama_model: Optional[str] = None, hf_token: Optional[str] = None):
        self.use_llama = use_llama and HAVE_TRANSFORMERS
        self.llama_model = llama_model
        self.llm = None
        self.hf_token = hf_token or None

        if self.use_llama:
            try:
                # build a text-generation pipeline; model name must be available locally or via HF
                kwargs = {}
                if self.hf_token:
                    kwargs['use_auth_token'] = self.hf_token
                self.llm = pipeline('text-generation', model=llama_model or 'meta-llama/Llama-2-7b-chat', **kwargs)
            except Exception as e:
                print(f"LLM pipeline couldn't be initialized: {e}; falling back to templates.")
                self.llm = None
                self.use_llama = False

    def load_fused_decision(self, path: str) -> Optional[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            print(f"Fused decision not found: {path}")
            return None
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"Failed to read fused decision: {e}")
            return None

    def load_catalog(self, csv_path: str) -> List[Dict[str, str]]:
        out = []
        p = Path(csv_path)
        if not p.exists():
            print(f"Meditation catalog not found at {csv_path}")
            return out
        with p.open('r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                out.append(r)
        return out

    def _find_catalog_entry(self, catalog: List[Dict[str, str]], name: str) -> Optional[Dict[str, str]]:
        if not name:
            return None
        name_norm = name.strip().lower()
        for e in catalog:
            if e.get('Name', '').strip().lower() == name_norm:
                return e
        # try partial match
        for e in catalog:
            if name_norm in e.get('Name', '').strip().lower():
                return e
        return None

    def _make_prompt(self, med_entry: Dict[str, str], fused: Dict[str, Any]) -> str:
        # Build a concise prompt for the LLM to generate tailored script / steps
        desc = med_entry.get('Description', '')
        base_instr = med_entry.get('Instructions', '')
        msm = fused.get('inputs', {}).get('msm', {})
        arm = fused.get('inputs', {}).get('arm', {})
        quality = fused.get('quality', {})

        prompt = (
            f"You are a helpful meditation coach. Produce a structured meditation session plan.\n\n"
            f"Meditation type: {med_entry.get('Name')}\n"
            f"Description: {desc}\n"
            f"Base instructions: {base_instr}\n\n"
            f"Signals from decision manager:\n"
            f"- selector label: {msm.get('label')} (confidence={msm.get('confidence')})\n"
            f"- arm/confidence: {arm.get('confidence')}\n"
            f"- quality: {quality.get('overall_ok')} notes: {quality.get('models', {})}\n\n"
            "Produce:\n"
            "1) A short tailored script (one paragraph)\n"
            "2) 4-6 step-by-step guidance bullets suitable for the user\n"
            "3) A recommended duration in minutes (short/medium/long) and a reason\n"
            "4) A difficulty level (beginner/intermediate/advanced) and why\n"
            "Keep the answer JSON-serializable. Use plain text."
        )
        return prompt

    def generate_plan(self, fused_path: str, catalog_csv: str, out_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        fused = self.load_fused_decision(fused_path)
        if fused is None:
            return None

        catalog = self.load_catalog(catalog_csv)

        msm_decision = fused.get('inputs', {}).get('msm', {})
        recommended_type = msm_decision.get('label') or fused.get('decision', {}).get('recommended_meditation')
        confidence = float(msm_decision.get('confidence', 0.0) or 0.0)

        entry = self._find_catalog_entry(catalog, recommended_type) if catalog else None

        # Duration logic
        if confidence > 0.8:
            duration_key = random.choice(['medium', 'long'])
        elif confidence > 0.5:
            duration_key = 'medium'
        else:
            duration_key = 'short'

        duration_map = {'short': 7, 'medium': 15, 'long': 25}
        duration_minutes = duration_map.get(duration_key, 10)

        difficulty_key = 'intermediate'
        # If quality overall_ok False, prefer beginner or reduce difficulty
        if fused.get('quality', {}).get('overall_ok') is False:
            difficulty_key = 'beginner'

        # If LLM available and enabled, call it to generate script/steps
        script = None
        steps = None
        llm_used = False

        if self.use_llama and self.llm is not None and entry is not None:
            prompt = self._make_prompt(entry, fused)
            try:
                # keep it short — request single completion
                resp = self.llm(prompt, max_length=512, do_sample=False)
                if isinstance(resp, list):
                    text = resp[0].get('generated_text') if isinstance(resp[0], dict) else str(resp[0])
                else:
                    text = str(resp)
                # simple split: first paragraph -> script, following lines bullets
                parts = [p.strip() for p in text.split('\n\n') if p.strip()]
                if parts:
                    script = parts[0]
                    # gather bullets
                    bullets = []
                    for p in parts[1:]:
                        for line in p.split('\n'):
                            line = line.strip('-* \t')
                            if line:
                                bullets.append(line)
                    steps = bullets[:6] if bullets else None
                    llm_used = True
            except Exception as e:
                print(f"LLM invocation failed: {e}; falling back to templates.")

        # Fallback: use CSV instructions or simple templates
        if script is None:
            if entry:
                # use Description + Instructions as script
                script = entry.get('Instructions') or entry.get('Description') or f"Practice {entry.get('Name')} mindfully."
            else:
                script = f"Practice {recommended_type or 'Mindfulness Meditation'} focusing on the breath and body."

        if steps is None:
            steps = [
                "Find a comfortable posture and settle for a few breaths.",
                f"Set your intention and focus on the core practice for {duration_minutes} minutes.",
                "If your attention wanders, gently bring it back without judgment.",
                "Finish by taking a few breaths and opening your eyes slowly."
            ]

        plan = {
            'recommended_meditation': entry.get('Name') if entry else (recommended_type or 'Mindfulness Meditation'),
            'confidence': confidence,
            'llm_used': llm_used,
            'plan_details': {
                'duration_minutes': duration_minutes,
                'duration_key': duration_key,
                'difficulty': difficulty_key,
                'script': script,
                'steps': steps,
            },
            'source_catalog_entry': entry,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        if out_path:
            try:
                p = Path(out_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
                print(f"Wrote meditation plan to {out_path}")
            except Exception as e:
                print(f"Failed to write plan to {out_path}: {e}")

        return plan


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate meditation recommendation plan from fused decision')
    parser.add_argument('--fused', type=str, default='preprocess_output/fused_decision.json', help='Path to fused_decision.json')
    parser.add_argument('--catalog', type=str, default='Core_engine/meditation.csv', help='Path to meditation.csv')
    parser.add_argument('--out', type=str, default='Core_engine/meditation_plan.json', help='Output JSON for the plan')
    parser.add_argument('--use-llama', action='store_true', help='Try to use transformers/LLama if installed')
    parser.add_argument('--llama-model', type=str, default=None, help='Transformers model name for LLM (optional)')
    parser.add_argument('--hf-token', type=str, default=None, help='Hugging Face token (or set HUGGINGFACE_HUB_TOKEN env var)')

    args = parser.parse_args()

    import os
    hf_token = args.hf_token or os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_HOME')
    mr = MeditationRecommender(use_llama=args.use_llama, llama_model=args.llama_model, hf_token=hf_token)
    plan = mr.generate_plan(args.fused, args.catalog, out_path=args.out)
    if plan:
        print(json.dumps(plan, indent=2, ensure_ascii=False))