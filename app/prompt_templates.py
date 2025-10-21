# app/prompt_templates.py


FEW_SHOT_EXAMPLES = [
{
"text": "Government passes new tax reform to help small businesses.",
"label": "REAL",
"explanation": "Reliable reporting, cites official finance ministry release.",
"sources": ["https://example.com/official-release"]
},
{
"text": "Aliens land in downtown Manhattan, government confirms.",
"label": "FAKE",
"explanation": "No credible sources; sensational wording and no official release.",
"sources": []
}
]


PROMPT_TEMPLATE = '''
You are a news verifier. Given a short news text, answer whether the news is REAL or FAKE.
Return a JSON object only (no extra commentary) with the following keys:
- label: "REAL" or "FAKE"
- confidence: a float between 0.0 and 1.0
- explanation: one-sentence reasoning
- sources: list of up to 3 URLs (strings). If none found, return an empty list.


Examples:
{few_shot_examples}


Now evaluate this news item and respond with JSON only:


News: "{news_text}"


Output (JSON):
'''.strip()




def build_prompt(news_text: str, use_few_shot: bool = True) -> str:
    few_shot = ''
    if use_few_shot:
        # include a concise few-shot section
        import json
        few_shot_list = [
        {"text": e["text"], "label": e["label"], "explanation": e["explanation"], "sources": e["sources"]}
        for e in FEW_SHOT_EXAMPLES
        ]
        few_shot = json.dumps(few_shot_list, indent=2)
    return PROMPT_TEMPLATE.format(few_shot_examples=few_shot, news_text=news_text)