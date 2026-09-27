import csv
import json
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "Qwen/Qwen3-4B-GGUF:Q4_K_M"

PROMPTS = {
    "en": (
        "Classify the expected short-term effect of the financial statement "
        "on the company's stock price, based only on the information provided. "
        "Return exactly one letter: "
        "A = strongly negative, B = negative, C = neutral or unclear, "
        "D = positive, E = strongly positive."
    ),

    "es": (
        "Clasifica el efecto esperado a corto plazo de la información financiera "
        "sobre el precio de las acciones de la empresa, basándote únicamente en "
        "la información proporcionada. Devuelve exactamente una letra: "
        "A = muy negativo, B = negativo, C = neutral o incierto, "
        "D = positivo, E = muy positivo."
    ),

    "de": (
        "Klassifiziere die erwartete kurzfristige Auswirkung der Finanzinformation "
        "auf den Aktienkurs des Unternehmens ausschließlich anhand der angegebenen "
        "Informationen. Gib genau einen Buchstaben zurück: "
        "A = stark negativ, B = negativ, C = neutral oder unklar, "
        "D = positiv, E = stark positiv."
    ),

    "fr": (
        "Classe l'effet attendu à court terme de l'information financière sur "
        "le cours de l'action de l'entreprise en te fondant uniquement sur les "
        "informations fournies. Réponds avec exactement une lettre : "
        "A = très négatif, B = négatif, C = neutre ou incertain, "
        "D = positif, E = très positif."
    ),

    "zh": (
        "仅根据所提供的财务信息，判断该信息对公司股价的预期短期影响。"
        "只返回一个字母：A = 强烈负面，B = 负面，C = 中性或不明确，"
        "D = 正面，E = 强烈正面。"
    ),

    "ja": (
        "提供された財務情報のみに基づいて、その情報が会社の株価に与える"
        "短期的な影響を分類してください。次のうち1文字だけを返してください："
        "A = 非常にネガティブ、B = ネガティブ、C = 中立または不明確、"
        "D = ポジティブ、E = 非常にポジティブ。"
    ),

    "ar": (
        "صنّف التأثير المتوقع على المدى القصير للمعلومة المالية على سعر سهم الشركة، "
        "بالاعتماد فقط على المعلومات المقدمة. أعد حرفًا واحدًا فقط: "
        "A = سلبي جدًا، B = سلبي، C = محايد أو غير واضح، "
        "D = إيجابي، E = إيجابي جدًا."
    ),
}

TESTS = [
    ("en", "The company reported substantially higher profit and raised its annual earnings outlook.", {"D", "E"}),
    ("en", "The company reported a substantially larger loss and lowered its annual earnings outlook.", {"A", "B"}),

    ("es", "La empresa informó de un fuerte aumento de los beneficios y elevó sus previsiones anuales.", {"D", "E"}),
    ("es", "La empresa informó de una fuerte caída de los beneficios y redujo sus previsiones anuales.", {"A", "B"}),

    ("de", "Das Unternehmen meldete einen deutlichen Gewinnanstieg und hob seine Jahresprognose an.", {"D", "E"}),
    ("de", "Das Unternehmen meldete einen deutlichen Gewinnrückgang und senkte seine Jahresprognose.", {"A", "B"}),

    ("fr", "L'entreprise a annoncé une forte hausse de ses bénéfices et a relevé ses prévisions annuelles.", {"D", "E"}),
    ("fr", "L'entreprise a annoncé une forte baisse de ses bénéfices et a abaissé ses prévisions annuelles.", {"A", "B"}),

    ("zh", "该公司利润大幅增长，并上调了全年盈利预期。", {"D", "E"}),
    ("zh", "该公司利润大幅下降，并下调了全年盈利预期。", {"A", "B"}),

    ("ja", "同社の利益は大幅に増加し、通期利益予想も引き上げられた。", {"D", "E"}),
    ("ja", "同社の利益は大幅に減少し、通期利益予想も引き下げられた。", {"A", "B"}),

    ("ar", "أعلنت الشركة عن ارتفاع كبير في الأرباح ورفعت توقعاتها السنوية للأرباح.", {"D", "E"}),
    ("ar", "أعلنت الشركة عن انخفاض كبير في الأرباح وخفضت توقعاتها السنوية للأرباح.", {"A", "B"}),
]


def classify(language, text):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROMPTS[language]},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 4,
    }

    request = urllib.request.Request(
        URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    elapsed = time.perf_counter() - start
    raw = result["choices"][0]["message"]["content"].strip()
    label = raw[:1].upper() if raw else ""

    return raw, label, elapsed


def main():
    results = []

    print("=" * 72)
    print("QWEN3-4B MULTILINGUAL SANITY CHECK V2")
    print("=" * 72)

    total_start = time.perf_counter()

    for language, text, expected in TESTS:
        raw, label, elapsed = classify(language, text)

        valid_format = label in {"A", "B", "C", "D", "E"}
        correct_direction = label in expected

        results.append({
            "language": language,
            "text": text,
            "raw_output": raw,
            "label": label,
            "expected": "/".join(sorted(expected)),
            "valid_format": valid_format,
            "correct_direction": correct_direction,
            "seconds": round(elapsed, 3),
        })

        print(
            f"{language.upper():2} | "
            f"output={raw!r:5} | "
            f"time={elapsed:5.2f}s | "
            f"{'PASS' if correct_direction else 'FAIL'}"
        )

    total = time.perf_counter() - total_start

    correct = sum(r["correct_direction"] for r in results)
    valid = sum(r["valid_format"] for r in results)

    Path("results").mkdir(exist_ok=True)

    out = Path("results/sanity_qwen3_4b_v2.csv")

    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    avg = total / len(results)

    print()
    print("=" * 72)
    print(f"Correct direction: {correct}/14")
    print(f"Valid A-E format:  {valid}/14")
    print(f"Average runtime:   {avg:.2f}s")
    print(f"Estimated 840:     {avg * 840 / 3600:.2f} hours")

    passed = correct >= 12 and valid == 14

    print()
    print("TECHNICAL GATE:", "PASS" if passed else "FAIL")
    print("=" * 72)


if __name__ == "__main__":
    main()