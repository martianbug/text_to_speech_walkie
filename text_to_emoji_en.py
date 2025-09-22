from typing import List, Tuple, Dict
import spacy
from collections import Counter
import emoji
import re

nlp = spacy.load("en_core_web_sm")
EMOJI_MAP: Dict[str, str] = {
    # --- Feelings / people ---
    "love": "❤️", "happy": "😄", "joy": "😄", "sad": "😢", "cry": "😢",
    "laugh": "😂", "surprise": "😲", "angry": "😠", "party": "🎉",
    "celebrate": "🎉", "celebration": "🎉",

    # --- Festivals & Holidays ---
    "festival": "🎊", "concert": "🎤", "music": "🎵", "band": "🎸",
    "dj": "🎧", "dance": "💃", "parade": "🥁",
    "fireworks": "🎆", "lantern": "🏮",
    "halloween": "🎃", "pumpkin": "🎃", "ghost": "👻",
    "christmas": "🎄", "xmas": "🎄", "santa": "🎅",
    "easter": "🐰", "egg": "🥚",
    "thanksgiving": "🦃", "turkey": "🦃",
    "new year": "🎆", "countdown": "⏳",
    "valentine": "💘",
    "wedding": "💒", "marriage": "💍",
    "birthday": "🎂", "anniversary": "💞",
    "carnival": "🎭", "masquerade": "🎭",
    "oktoberfest": "🍺", "beer": "🍺",
    "diwali": "🪔",
    "hanukkah": "🕎",
    "ramadan": "🕌", "eid": "🕌",
    "festival lights": "🪔",
    "independence": "🎆",

    # --- Technology / Digital life ---
    "computer": "💻", "laptop": "💻", "desktop": "🖥️",
    "smartphone": "📱", "phone": "📱", "tablet": "📱",
    "camera": "📷", "photo": "📸", "selfie": "🤳",
    "video": "🎥", "stream": "📺", "livestream": "📺",
    "game": "🎮", "gamer": "🎮", "gaming": "🎮",
    "virtual reality": "🕶️", "vr": "🕶️",
    "augmented reality": "🕶️",
    "robot": "🤖", "ai": "🤖",
    "code": "💻", "programming": "💻", "developer": "👩‍💻",
    "keyboard": "⌨️", "mouse": "🖱️",
    "server": "🖥️", "cloud": "☁️",
    "internet": "🌐", "web": "🌐", "website": "🌐",
    "wifi": "📶", "network": "📡",
    "email": "✉️", "message": "✉️", "chat": "💬",
    "call": "📞", "video call": "📹",
    "social": "💬", "social media": "💬",
    "streaming": "🎬", "music app": "🎵",
    "crypto": "₿", "bitcoin": "₿", "blockchain": "🔗",
    "database": "🗄️",
    "printer": "🖨️",
    "headphones": "🎧", "earbuds": "🎧",
    "drone": "🚁",
    "camera drone": "🚁",
    "watch": "⌚", "smartwatch": "⌚",
    "microphone": "🎙️",
    "usb": "🔌", "charger": "🔌",
    "energy": "⚡", "battery": "🔋",

    # --- Other general categories for completeness ---
    "travel": "🌍", "plane": "✈️", "airport": "🛫",
    "train": "🚆", "car": "🚗", "bus": "🚌",
    "food": "🍽️", "pizza": "🍕", "burger": "🍔",
    "coffee": "☕", "tea": "🍵",
    "sun": "☀️", "rain": "🌧️", "snow": "❄️",
    "work": "💼", "study": "📚",
    "money": "💰", "pay": "💳",
}

def normalize_token(tok: str) -> str:
    tok = tok.lower().strip()
    tok = re.sub(r"[^\w']+", "", tok)  # keep apostrophes for contractions
    return tok

def extract_keywords(text: str, topn: int = 5) -> List[Tuple[str, float]]:
    """
    Extract candidate keywords using spaCy (EN).
    Consider NOUN, PROPN, VERB, ADJ. Use lemmas and noun chunks.
    Returns list of (lemma, score) ordered by score desc.
    """
    doc = nlp(text)
    candidates = []

    for token in doc:
        if token.is_stop or token.is_punct or token.like_num:
            continue
        if token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}:
            lemma = normalize_token(token.lemma_)
            if lemma:
                candidates.append(lemma)

    # noun chunks (boost)
    for nc in doc.noun_chunks:
        lemma = normalize_token(nc.root.lemma_)
        if lemma:
            candidates.extend([lemma] * 2)

    counts = Counter(candidates)
    if not counts:
        return []

    most = counts.most_common(topn)
    max_cnt = most[0][1]
    return [(kw, cnt / max_cnt) for kw, cnt in most]

def map_keywords_to_emojis(keywords: List[Tuple[str, float]], max_emojis: int = 5) -> List[str]:
    chosen = []
    seen = set()
    for lemma, score in keywords:
        if lemma in EMOJI_MAP:
            e = EMOJI_MAP[lemma]
            if e not in seen:
                chosen.append(e); seen.add(e)
            if len(chosen) >= max_emojis: break
            continue
        # substring match
        found = False
        for key, em in EMOJI_MAP.items():
            if lemma in key or key in lemma:
                if em not in seen:
                    chosen.append(em); seen.add(em)
                found = True
                break
        if found and len(chosen) >= max_emojis: break
    return chosen

def text_to_emoji_summary(text: str, max_keywords: int = 6, max_emojis: int = 5) -> str:
    kws = extract_keywords(text, topn=max_keywords)
    emojis = map_keywords_to_emojis(kws, max_emojis=max_emojis)
    if emojis:
        return " ".join(emojis)
    return " / ".join([kw for kw, _ in kws]) or ""

if __name__ == "__main__":
    examples = [
        "I went to the beach with my dog, we ate pizza and drank beer. It was a perfect day.",
        "I have a work meeting at 10 AM, I need to prepare the presentation and bring my laptop.",
        "I'm very tired, I want to sleep and dream about traveling."
    ]
    for t in examples:
        print("Text:", t)
        print("Emojis:", text_to_emoji_summary(t))
        print("---")
