import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PRIMARY_RUNNER = ROOT / "run_inference.py"
PRIMARY_PROMPTS = ROOT / "config" / "prompts.json"

ROBUSTNESS_CONFIG = ROOT / "config" / "robustness"

EXPECTED_PRIMARY_PROMPTS_SHA256 = (
    "8d117ae66128f87d3dd2ddc97a178fdeb87f88046ef1e4ec6c9c09763a17588b"
)


R2_PROMPTS = {
    "en": (
        "Classify the expected short-term effect of the financial statement "
        "on the company's stock price, based only on the information provided. "
        "Use the following scale: "
        "A = strongly negative, B = negative, C = neutral or unclear, "
        "D = positive, E = strongly positive. "
        "Output exactly ONE uppercase letter: A, B, C, D, or E. "
        "Do not output words, punctuation, explanations, or any additional text."
    ),

    "es": (
        "Clasifica el efecto esperado a corto plazo de la información financiera "
        "sobre el precio de las acciones de la empresa, basándote únicamente en "
        "la información proporcionada. "
        "Usa la siguiente escala: "
        "A = muy negativo, B = negativo, C = neutral o incierto, "
        "D = positivo, E = muy positivo. "
        "Devuelve exactamente UNA letra mayúscula: A, B, C, D o E. "
        "No devuelvas palabras, signos de puntuación, explicaciones ni texto adicional."
    ),

    "de": (
        "Klassifiziere die erwartete kurzfristige Auswirkung der Finanzinformation "
        "auf den Aktienkurs des Unternehmens ausschließlich anhand der "
        "bereitgestellten Informationen. "
        "Verwende die folgende Skala: "
        "A = stark negativ, B = negativ, C = neutral oder unklar, "
        "D = positiv, E = stark positiv. "
        "Gib genau EINEN Großbuchstaben zurück: A, B, C, D oder E. "
        "Gib keine Wörter, Satzzeichen, Erklärungen oder zusätzlichen Text aus."
    ),

    "fr": (
        "Classe l'effet attendu à court terme de l'information financière sur "
        "le cours de l'action de l'entreprise en te fondant uniquement sur les "
        "informations fournies. "
        "Utilise l'échelle suivante : "
        "A = très négatif, B = négatif, C = neutre ou incertain, "
        "D = positif, E = très positif. "
        "Réponds avec exactement UNE lettre majuscule : A, B, C, D ou E. "
        "Ne fournis aucun mot, signe de ponctuation, explication ou texte supplémentaire."
    ),

    "zh": (
        "仅根据所提供的财务信息，判断该信息对公司股价的预期短期影响。"
        "使用以下等级："
        "A = 强烈负面，B = 负面，C = 中性或不明确，"
        "D = 正面，E = 强烈正面。"
        "只输出一个大写字母：A、B、C、D 或 E。"
        "不要输出任何单词、标点、解释或其他文字。"
    ),

    "ja": (
        "提供された財務情報のみに基づいて、その情報が会社の株価に与える"
        "短期的な影響を分類してください。"
        "次の尺度を使用してください："
        "A = 非常にネガティブ、B = ネガティブ、C = 中立または不明確、"
        "D = ポジティブ、E = 非常にポジティブ。"
        "A、B、C、D、E のいずれか大文字1文字だけを出力してください。"
        "単語、句読点、説明、その他の文章は出力しないでください。"
    ),

    "ar": (
        "صنّف التأثير المتوقع على المدى القصير للمعلومة المالية على سعر سهم الشركة، "
        "بالاعتماد فقط على المعلومات المقدمة. "
        "استخدم المقياس التالي: "
        "A = سلبي جدًا، B = سلبي، C = محايد أو غير واضح، "
        "D = إيجابي، E = إيجابي جدًا. "
        "أعد حرفًا إنجليزيًا كبيرًا واحدًا فقط: A أو B أو C أو D أو E. "
        "لا تُخرج كلمات أو علامات ترقيم أو تفسيرات أو أي نص إضافي."
    ),
}


R3_PROMPTS = {
    "en": (
        "Based solely on the financial statement provided, assess its likely "
        "short-term impact on the company's share price. "
        "Choose one category: "
        "A = strongly negative, B = negative, C = neutral or uncertain, "
        "D = positive, E = strongly positive. "
        "Respond only with the corresponding letter."
    ),

    "es": (
        "Basándote exclusivamente en la información financiera proporcionada, "
        "evalúa su probable impacto a corto plazo sobre el precio de las acciones "
        "de la empresa. "
        "Elige una categoría: "
        "A = muy negativo, B = negativo, C = neutral o incierto, "
        "D = positivo, E = muy positivo. "
        "Responde únicamente con la letra correspondiente."
    ),

    "de": (
        "Beurteile ausschließlich anhand der bereitgestellten Finanzinformation "
        "deren wahrscheinliche kurzfristige Auswirkung auf den Aktienkurs des "
        "Unternehmens. "
        "Wähle eine Kategorie: "
        "A = stark negativ, B = negativ, C = neutral oder ungewiss, "
        "D = positiv, E = stark positiv. "
        "Antworte ausschließlich mit dem entsprechenden Buchstaben."
    ),

    "fr": (
        "En te fondant exclusivement sur l'information financière fournie, "
        "évalue son impact probable à court terme sur le cours de l'action "
        "de l'entreprise. "
        "Choisis une catégorie : "
        "A = très négatif, B = négatif, C = neutre ou incertain, "
        "D = positif, E = très positif. "
        "Réponds uniquement avec la lettre correspondante."
    ),

    "zh": (
        "仅根据所提供的财务信息，评估其对公司股价可能产生的短期影响。"
        "请选择一个类别："
        "A = 强烈负面，B = 负面，C = 中性或不确定，"
        "D = 正面，E = 强烈正面。"
        "只回答相应的字母。"
    ),

    "ja": (
        "提供された財務情報だけに基づいて、それが会社の株価に与える"
        "短期的な影響を評価してください。"
        "次のカテゴリーから1つ選んでください："
        "A = 非常にネガティブ、B = ネガティブ、C = 中立または不確実、"
        "D = ポジティブ、E = 非常にポジティブ。"
        "対応する文字だけで回答してください。"
    ),

    "ar": (
        "استنادًا فقط إلى المعلومة المالية المقدمة، قيّم تأثيرها المحتمل "
        "قصير الأجل على سعر سهم الشركة. "
        "اختر فئة واحدة: "
        "A = سلبي جدًا، B = سلبي، C = محايد أو غير مؤكد، "
        "D = إيجابي، E = إيجابي جدًا. "
        "أجب بالحرف المقابل فقط."
    ),
}


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def write_json(path, data):
    text = (
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    return sha256_file(path)


def make_runner(
    prompt_filename,
    prompt_hash,
    output_filename,
    runner_filename,
    title,
):
    text = PRIMARY_RUNNER.read_text(
        encoding="utf-8"
    )

    text = text.replace(
        'PROMPTS_PATH = ROOT / "config" / "prompts.json"',
        f'PROMPTS_PATH = ROOT / "config" / "robustness" / "{prompt_filename}"',
    )

    text = re.sub(
        r'EXPECTED_PROMPTS_SHA256\s*=\s*\(\s*"[^"]+"\s*\)',
        (
            "EXPECTED_PROMPTS_SHA256 = (\n"
            f'    "{prompt_hash}"\n'
            ")"
        ),
        text,
        count=1,
    )

    text = text.replace(
        'OUTPUT_PATH = ROOT / "results" / "main_inference_raw.csv"',
        (
            'OUTPUT_PATH = ROOT / "results" / '
            f'"robustness" / "{output_filename}"'
        ),
    )

    text = text.replace(
        '"max_tokens": 4,',
        '"max_tokens": 12,',
    )

    text = text.replace(
        "MAIN INFERENCE — FROZEN EXPERIMENT",
        title,
    )

    runner_path = ROOT / runner_filename

    runner_path.write_text(
        text,
        encoding="utf-8",
    )

    return runner_path


def main():
    print("=" * 72)
    print("PREPARE ROBUSTNESS EXPERIMENTS R2 + R3")
    print("=" * 72)

    if not PRIMARY_RUNNER.exists():
        raise RuntimeError(
            f"Missing {PRIMARY_RUNNER}"
        )

    if not PRIMARY_PROMPTS.exists():
        raise RuntimeError(
            f"Missing {PRIMARY_PROMPTS}"
        )

    primary_hash = sha256_file(
        PRIMARY_PROMPTS
    )

    if primary_hash != EXPECTED_PRIMARY_PROMPTS_SHA256:
        raise RuntimeError(
            "Primary prompts changed.\n"
            f"Expected: {EXPECTED_PRIMARY_PROMPTS_SHA256}\n"
            f"Actual:   {primary_hash}"
        )

    print("Primary prompts hash: PASS")

    ROBUSTNESS_CONFIG.mkdir(
        parents=True,
        exist_ok=True,
    )

    r2_prompt_path = (
        ROBUSTNESS_CONFIG
        / "prompts_r2.json"
    )

    r3_prompt_path = (
        ROBUSTNESS_CONFIG
        / "prompts_r3.json"
    )

    r2_hash = write_json(
        r2_prompt_path,
        R2_PROMPTS,
    )

    r3_hash = write_json(
        r3_prompt_path,
        R3_PROMPTS,
    )

    (
        ROBUSTNESS_CONFIG
        / "prompts_r2.sha256"
    ).write_text(
        f"{r2_hash}  config/robustness/prompts_r2.json\n",
        encoding="ascii",
    )

    (
        ROBUSTNESS_CONFIG
        / "prompts_r3.sha256"
    ).write_text(
        f"{r3_hash}  config/robustness/prompts_r3.json\n",
        encoding="ascii",
    )

    r2_spec = {
        "run_id": "r2_strict_prompt",
        "type": "post-hoc robustness check",
        "purpose": (
            "Test sensitivity to stronger output-format instructions "
            "while holding max_tokens at 12."
        ),
        "comparison": "R2 vs R1 isolates prompt-format wording.",
        "max_tokens": 12,
        "temperature": 0,
        "reasoning": False,
        "model": "Qwen/Qwen3-4B-GGUF:Q4_K_M",
        "prompts_sha256": r2_hash,
    }

    r3_spec = {
        "run_id": "r3_paraphrase",
        "type": "post-hoc robustness check",
        "purpose": (
            "Test sensitivity to a semantically equivalent prompt "
            "paraphrase while holding max_tokens at 12."
        ),
        "comparison": "R3 vs R1 isolates prompt wording.",
        "max_tokens": 12,
        "temperature": 0,
        "reasoning": False,
        "model": "Qwen/Qwen3-4B-GGUF:Q4_K_M",
        "prompts_sha256": r3_hash,
    }

    write_json(
        ROBUSTNESS_CONFIG
        / "r2_strict_prompt.json",
        r2_spec,
    )

    write_json(
        ROBUSTNESS_CONFIG
        / "r3_paraphrase.json",
        r3_spec,
    )

    r2_runner = make_runner(
        prompt_filename="prompts_r2.json",
        prompt_hash=r2_hash,
        output_filename="r2_strict_prompt_raw.csv",
        runner_filename="run_robustness_r2.py",
        title=(
            "ROBUSTNESS R2 — STRICT FORMAT PROMPT, "
            "MAX_TOKENS=12"
        ),
    )

    r3_runner = make_runner(
        prompt_filename="prompts_r3.json",
        prompt_hash=r3_hash,
        output_filename="r3_paraphrase_raw.csv",
        runner_filename="run_robustness_r3.py",
        title=(
            "ROBUSTNESS R3 — PARAPHRASED PROMPT, "
            "MAX_TOKENS=12"
        ),
    )

    print()
    print("R2 prompt SHA256:")
    print(r2_hash)

    print()
    print("R3 prompt SHA256:")
    print(r3_hash)

    print()
    print("Created:")
    print(r2_prompt_path.relative_to(ROOT))
    print(r3_prompt_path.relative_to(ROOT))
    print(r2_runner.relative_to(ROOT))
    print(r3_runner.relative_to(ROOT))

    print()
    print("PREPARATION COMPLETE — NO INFERENCE WAS RUN")


if __name__ == "__main__":
    main()