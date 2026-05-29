"""
Offline script: Build upi_fraud_phrases.json for Scorer F.

Creates 7 fraud cluster centroids using paraphrase-multilingual-MiniLM-L12-v2.
Each cluster has curated Hindi/Hinglish/English phrases that appear in UPI
transaction remarks for the corresponding fraud type.

Output: ml/models/upi_fraud_phrases.json
Format: {cluster_name: [[float, ...], ...]}  — one sub-list per phrase embedding.

Run: python ml/scripts/build_phrase_dict.py
     python ml/scripts/build_phrase_dict.py --output ml/models/custom_phrases.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Add repo root to path for ml imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


PHRASE_CLUSTERS: dict[str, list[str]] = {
    "digital_arrest_hindi": [
        "CBI arrest warrant",
        "police case registered",
        "FIR darj hogi",
        "digital arrest",
        "ED notice",
        "court case",
        "ghabraaiye mat payment karo",
        "sarkar ne notice bheja hai",
        "aapka account freeze hoga",
        "cybercrime investigation",
        "arrest se bachne ke liye",
        "turant payment karen",
        "warrant nikla hai aapke naam",
        "jail se bachne ke liye",
        "case band karne ke liye paisa bhejo",
    ],
    "investment_fraud_hindi": [
        "guaranteed return",
        "100 percent profit",
        "double your money",
        "stock tips ke liye",
        "share market se kamaao",
        "investment return",
        "profit withdrawal",
        "trading account deposit",
        "scheme mein invest karo",
        "paisa double hoga",
        "high returns guaranteed",
        "crypto investment",
        "forex trading",
        "bot se trading",
        "daily income scheme",
    ],
    "otp_social_eng_hindi": [
        "OTP share karo",
        "verification code batao",
        "bank se bol raha hoon",
        "KYC update ke liye OTP",
        "SIM card band hoga",
        "account verify karo",
        "ATM card block hoga",
        "ek baar OTP batao",
        "aapki identity verify karni hai",
        "account suspend mat hone do",
        "NPCI se baat kar raha hoon",
        "link par click karo",
        "aapka account hack ho gaya",
        "RBI ka notice aaya hai",
    ],
    "otp_social_eng_english": [
        "your OTP is",
        "share your OTP",
        "verify your account",
        "bank executive calling",
        "your KYC is pending",
        "account will be blocked",
        "click this link to verify",
        "your card will expire",
        "update your details",
        "NPCI verification required",
        "your UPI PIN",
        "send money to verify",
        "one time verification",
        "your account is compromised",
    ],
    "romance_scam_english": [
        "gift for you",
        "gift parcel stuck",
        "customs clearance",
        "release my parcel",
        "gift from abroad",
        "clearance fee",
        "transfer for love",
        "emergency abroad",
        "flight ticket stuck",
        "need help money",
        "please help me",
        "medical emergency abroad",
        "can you send me",
        "i trust you",
        "we will meet soon",
    ],
    "lottery_fraud_hindi": [
        "aap jeete hain",
        "lucky draw winner",
        "prize money release",
        "lottery jeet gaye",
        "scratch card winner",
        "crore jeeta hai",
        "prize amount transfer",
        "lucky winner selected",
        "tax bharo prize pao",
        "processing fee de do",
        "prize ke liye advance",
        "national lottery",
        "aapka number draw hua",
        "congratulations selected",
        "prize release karo",
    ],
    "sim_swap_indicators": [
        "SIM port karna hai",
        "number port karo",
        "SIM card upgrade",
        "new SIM activate",
        "OTP for porting",
        "transfer number",
        "SIM swap verification",
        "mobile number change",
        "sim blocked release",
        "port out request",
        "network change OTP",
        "jio se airtel port",
        "vi port request",
        "sim registration",
        "primary number change",
    ],
}

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "upi_fraud_phrases.json"
)


def build_phrase_dict(output_path: str) -> None:
    print("Loading sentence-transformers model (MiniLM)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    result: dict[str, list[list[float]]] = {}
    total_phrases = 0

    for cluster_name, phrases in PHRASE_CLUSTERS.items():
        print(f"  Encoding cluster '{cluster_name}' ({len(phrases)} phrases)...")
        embeddings = model.encode(
            phrases,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result[cluster_name] = embeddings.tolist()
        total_phrases += len(phrases)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {total_phrases} phrases across {len(result)} clusters.")
    print(f"Saved: {os.path.abspath(output_path)}")
    print("\nVerify Scorer F loads without error:")
    print("  python -c \"from app.detection.tier3 import scorer_f; scorer_f._load_scorer_f(); print('OK')\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build UPI fraud phrase embeddings for Scorer F")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output JSON path")
    args = parser.parse_args()
    build_phrase_dict(args.output)
