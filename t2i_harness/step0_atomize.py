"""
THEMIS T2I Step 0 - Prompt Atomization and Taxonomy Linking.

This module takes GenEval2 prompt data (VQA-format atoms) and produces a
structured atomized representation with:
  - Unified atom format (question, answer, answer_type, skill, target_object, weight)
  - Extracted objects with taxonomy context (class_id, class_name, diagnostic_checkpoints)
  - Per-object attributes and counts extracted from the prompt/atoms

The output is consumed by downstream T2I evaluation steps (router, experts, scoring).
"""

import os
import re
import sys
import json
from pathlib import Path
from openai import OpenAI

if __name__ != "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
T2I_DIR = Path(__file__).resolve().parent

# Ensure t2i_harness is importable for `from common import ...`
if str(T2I_DIR) not in sys.path:
    sys.path.insert(0, str(T2I_DIR))

from common import api_call_with_retry

TAXONOMY_DIR = PROJECT_ROOT / "taxonomy_info"
TAXONOMY_STRUCTURAL_DIR = PROJECT_ROOT / "taxonomy_info_structural"

ATOMIZE_MODEL = "qwen3.6-plus"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Number-word helpers (mirrors GenEval2 benchmark_schema.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NUMBER_WORDS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Taxonomy Loading (re-implemented from c2i_harness/step1_router.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_taxonomy_info(class_id: int) -> dict | None:
    """Read enriched taxonomy info for a given ImageNet class_id.

    Files are organized in batches of 10 class IDs:
        taxonomy_info/taxonomy_enriched_Batch_{class_id // 10}.json
    """
    batch_num = class_id // 10
    batch_file = TAXONOMY_DIR / f"taxonomy_enriched_Batch_{batch_num}.json"
    if not batch_file.exists():
        return None
    with open(batch_file, "r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if item.get("class_id") == class_id:
            return item
    return None


def get_structured_taxonomy_info(class_id: int) -> dict | None:
    """Read structured taxonomy info (diagnostic_checkpoints) for a given class_id.

    Files are organized in batches of 10 class IDs:
        taxonomy_info_structural/taxonomy_enriched_Batch_{class_id // 10}_structured.json
    """
    batch_num = class_id // 10
    batch_file = TAXONOMY_STRUCTURAL_DIR / f"taxonomy_enriched_Batch_{batch_num}_structured.json"
    if not batch_file.exists():
        return None
    with open(batch_file, "r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if item.get("class_id") == class_id:
            return item
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GenEval2 Data Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_geneval2_prompt(jsonl_path: str | Path, prompt_id: str | int) -> dict | None:
    """Load a specific prompt from a GenEval2 JSONL file by prompt_id.

    The prompt_id is the 0-based line index in the JSONL file.

    Args:
        jsonl_path: Path to the geneval2_data.jsonl file.
        prompt_id: The 0-based line index (int or str representation).

    Returns:
        The parsed JSON record (dict) for the requested prompt, or None if
        the file does not exist or the index is out of range.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"[WARN] GenEval2 JSONL not found: {jsonl_path}")
        return None

    target_index = int(prompt_id)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index == target_index:
                line = line.strip()
                if not line:
                    return None
                record = json.loads(line)
                record["prompt_id"] = str(index)
                return record
    print(f"[WARN] prompt_id {prompt_id} not found in {jsonl_path} (file has {index + 1} lines).")
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Answer Type Inference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _infer_answer_type(question: str, answer) -> str:
    """Infer the answer type from a question-answer pair.

    Types: "count" (numeric answers, "how many" questions),
           "binary" (yes/no), "string" (everything else).
    """
    answer_text = str(answer).strip()
    question_text = str(question).strip().lower()
    answer_lower = answer_text.lower()
    if question_text.startswith("how many"):
        return "count"
    if answer_lower in {"yes", "no"}:
        return "binary"
    if answer_lower in NUMBER_WORDS or answer_text.isdigit():
        return "count"
    return "string"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Object Extraction from VQA Questions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_object_from_question(question: str) -> str:
    """Extract the target object name from a GenEval2 VQA question.

    Recognized question patterns:
      - "How many {plural_object} are in the image?"
      - "Are there any {plural_object} in the image?"
      - "Are the {plural_object} {attribute}?"
      - "Is the {object} {attribute}?"

    Returns the (possibly plural) object name, or "" if no pattern matches.
    """
    q = question.strip()

    # Pattern 1: "How many {X} are in the image?"
    m = re.match(r"How many (.+?) are in the image\??", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Pattern 2: "Are there any {X} in the image?"
    m = re.match(r"Are there any (.+?) in the image\??", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Pattern 3: "Are the {X} {attr}?" or "Is the {X} {attr}?"
    m = re.match(r"(?:Are|Is) the (\w+) \w+\?", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return ""


def _singularize(word: str) -> str:
    """Best-effort singularization of common English nouns.

    Handles regular plurals, -ies/-y, -es after s/x/z/ch/sh, and common
    irregulars (sheep, deer, fish, etc.). Not exhaustive but covers the
    object names that appear in GenEval2 prompts.
    """
    word = word.lower().strip()

    # Irregular plurals (same singular and plural)
    irregulars = {
        "sheep", "deer", "fish", "moose", "species", "series",
        "aircraft", "salmon", "trout", "swine", "bison", "shrimp",
    }
    if word in irregulars:
        return word

    # "ies" -> "y"  (berries -> berry)
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"

    # "xes", "zes", "ches", "shes" -> remove "es"  (boxes -> box, watches -> watch)
    if word.endswith(("xes", "zes", "ches", "shes")):
        return word[:-2]

    # Remaining "s" (not "ss") -> remove "s"
    # This handles: monkeys -> monkey, horses -> horse, sheeps -> sheep
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Atom Normalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize_atoms(vqa_list: list, skills: list[str] | None = None) -> list[dict]:
    """Convert GenEval2 VQA format to unified atom format.

    Each GenEval2 VQA pair is a [question, answer] tuple. This function
    enriches each pair with answer_type, skill, target_object, and weight.

    Args:
        vqa_list: List of [question, answer] pairs from GenEval2.
        skills: Optional list of skill labels, parallel to vqa_list.
            If not provided or shorter than vqa_list, missing entries
            default to "unspecified".

    Returns:
        List of atom dicts, each with keys:
            atom_index, question, answer, expected, answer_type, skill,
            target_object, weight
        ("expected" is an alias for "answer", used by Router/Judge prompts.)
    """
    if skills is None:
        skills = []
    atoms: list[dict] = []
    for index, pair in enumerate(vqa_list):
        question, answer = pair
        skill = skills[index] if index < len(skills) else "unspecified"
        target_object = _extract_object_from_question(question)
        answer_type = _infer_answer_type(question, answer)
        atoms.append({
            "atom_index": index,
            "question": question,
            "answer": answer,
            "expected": answer,
            "answer_type": answer_type,
            "skill": skill,
            "target_object": target_object,
            "weight": 1.0,
        })
    return atoms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Object Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_objects_from_atoms(atoms: list[dict]) -> list[str]:
    """Extract unique singularized object names from a list of atoms.

    Args:
        atoms: List of atom dicts (output of normalize_atoms).

    Returns:
        Sorted list of unique singular object names.
    """
    objects: set[str] = set()
    for atom in atoms:
        target = atom.get("target_object", "")
        if target:
            objects.add(_singularize(target))
    return sorted(objects)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Object-to-ClassID Mapping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Hardcoded mapping from common object names to ImageNet class IDs.
# These are best-effort mappings for objects that frequently appear in
# GenEval2 prompts. Objects not in this map will have class_id=None and
# no taxonomy context (the atomization still succeeds; experts will work
# without prior taxonomy knowledge).
#
# ImageNet class IDs are approximate; some objects (e.g., "croissant",
# "trumpet") may not have exact ImageNet equivalents and use the closest
# available class.
_COMMON_OBJECT_TO_CLASSID: dict[str, int] = {
    # --- Primates ---
    "monkey": 376,          # guenon
    "baboon": 379,          # baboon
    "chimpanzee": 367,      # chimpanzee / chimp
    "chimp": 367,
    "gorilla": 366,         # gorilla
    "gibbon": 368,          # gibbon, Hylobates lar
    "orangutan": 374,       # orangutan / siamang

    # --- Canines & felines ---
    "dog": 235,             # German shepherd (representative)
    "puppy": 235,
    "poodle": 265,          # standard poodle
    "cat": 281,             # tabby cat
    "kitten": 281,
    "lion": 291,            # lion
    "tiger": 292,           # tiger
    "fox": 277,             # red fox
    "leopard": 288,         # leopard
    "cheetah": 293,         # cheetah
    "wolf": 270,            # timber wolf, gray wolf

    # --- Large mammals ---
    "elephant": 385,        # Indian elephant
    "zebra": 340,           # zebra
    "bear": 294,            # brown bear
    "sheep": 348,           # ram / domestic sheep (closest)
    "ram": 348,
    "cow": 345,             # ox
    "ox": 345,
    "pig": 341,             # pig / hog
    "hog": 341,
    "horse": 339,           # sorrel
    "giraffe": 350,         # gazelle / closest large herbivore
    "kangaroo": 104,        # wallaby (closest)
    "deer": 351,            # hartebeest (closest)
    "raccoon": 358,         # raccoon
    "rabbit": 331,          # wood rabbit, hare
    "hare": 331,
    "hamster": 333,         # hamster
    "squirrel": 335,        # fox squirrel
    "camel": 354,           # Arabian camel / dromedary
    "panda": 388,           # giant panda

    # --- Birds ---
    "bird": 7,              # cock / rooster
    "rooster": 7,
    "parrot": 85,           # African grey / macaw
    "duck": 97,             # drake
    "owl": 24,              # great grey owl
    "penguin": 145,         # king penguin
    "flamingo": 130,        # flamingo
    "peacock": 84,          # peacock / peahen

    # --- Small animals ---
    "fish": 1,              # goldfish (representative)
    "frog": 30,             # bullfrog
    "turtle": 36,           # mud turtle, terrapin
    "butterfly": 323,       # monarch butterfly
    "spider": 72,           # garden spider
    "snake": 54,            # green snake / snake
    "lizard": 49,           # green lizard / American chameleon

    # --- Vehicles ---
    "bicycle": 444,         # bicycle
    "bike": 444,
    "car": 511,             # convertible (representative)
    "truck": 717,           # pickup truck
    "motorcycle": 670,      # motor scooter
    "bus": 654,             # minibus
    "ambulance": 408,       # ambulance
    "train": 466,           # bullet train / high-speed train
    "airplane": 404,        # airliner / passenger plane
    "boat": 472,           # canoe / closest watercraft
    "ship": 472,

    # --- Everyday objects ---
    "backpack": 412,        # backpack, back pack
    "umbrella": 814,        # umbrella (parasol)
    "clock": 892,           # wall clock
    "trumpet": 558,         # French horn (closest brass instrument)
    "donut": 596,           # doughnut, donut
    "doughnut": 596,
    "bagel": 931,           # bagel
    "flower": 985,          # daisy
    "daisy": 985,
    "rose": 949,            # rose (not exact, but close floral)
    "croissant": 925,       # closest food item
    "pretzel": 930,         # pretzel
    "pizza": 963,           # pizza
    "banana": 954,          # banana
    "apple": 948,           # Granny Smith / apple
    "orange": 950,          # orange (fruit)
    "bottle": 898,          # water bottle / wine bottle
    "cup": 641,             # cup / coffee mug
    "book": 931,            # (closest; ImageNet has no direct "book")
    "chair": 559,           # folding chair
    "table": 532,           # dining table
    "guitar": 546,          # acoustic guitar / electric guitar
    "piano": 579,           # grand piano
}


def build_object_to_classid_map() -> dict[str, int]:
    """Build a mapping from common object names to ImageNet class IDs.

    Uses a hardcoded dictionary of common objects. If a JSON file exists at
    T2I_DIR/object_to_classid.json, it is loaded as an override/extension on
    top of the hardcoded defaults.

    Returns:
        Dict mapping object_name (singular, lowercase) to ImageNet class_id (int).
    """
    mapping = dict(_COMMON_OBJECT_TO_CLASSID)

    json_path = T2I_DIR / "object_to_classid.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            if isinstance(overrides, dict):
                for k, v in overrides.items():
                    try:
                        mapping[k.lower().strip()] = int(v)
                    except (ValueError, TypeError):
                        print(f"[WARN] Skipping invalid class_id for '{k}': {v}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Failed to load object_to_classid.json: {e}")

    return mapping


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Taxonomy Linking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def link_taxonomy(object_name: str, object_to_classid: dict[str, int]) -> dict | None:
    """Link an object name to its ImageNet class and load taxonomy context.

    Args:
        object_name: Singular object name (e.g., "monkey").
        object_to_classid: Mapping from object names to ImageNet class IDs
            (output of build_object_to_classid_map).

    Returns:
        Dict with keys: object_name, class_id, class_name, taxonomy_description,
        diagnostic_checkpoints. Returns None if the object is not in the
        mapping (no class_id available).
    """
    key = object_name.lower().strip()
    class_id = object_to_classid.get(key)
    if class_id is None:
        return None

    taxonomy_info = get_taxonomy_info(class_id)
    structured_info = get_structured_taxonomy_info(class_id)

    class_name = ""
    taxonomy_description = ""
    diagnostic_checkpoints: dict = {}

    if taxonomy_info:
        class_name = taxonomy_info.get("class_name", "")
        taxonomy_description = taxonomy_info.get("enriched_description", "")

    if structured_info:
        diagnostic_checkpoints = structured_info.get("diagnostic_checkpoints", {})

    return {
        "object_name": key,
        "class_id": class_id,
        "class_name": class_name,
        "taxonomy_description": taxonomy_description,
        "diagnostic_checkpoints": diagnostic_checkpoints,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Attribute and Count Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_attribute_from_question(question: str) -> str | None:
    """Extract the attribute word from an attribute-skill VQA question.

    Recognized patterns:
      - "Are the {object} {attribute}?"
      - "Is the {object} {attribute}?"

    Returns the attribute word (e.g., "brown" from "Are the monkeys brown?"),
    or None if the pattern does not match.
    """
    q = question.strip()
    m = re.match(r"(?:Are|Is) the \w+ (\w+)\?", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_count_from_answer(answer) -> int | None:
    """Convert a count answer to an integer.

    Handles number words ("four" -> 4), digit strings ("4" -> 4),
    and numeric values (4 -> 4). Returns None if conversion fails.
    """
    answer_str = str(answer).strip().lower()
    if answer_str in NUMBER_WORDS:
        return int(NUMBER_WORDS[answer_str])
    try:
        return int(answer_str)
    except (ValueError, TypeError):
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Atomization Function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def atomize_prompt(prompt_data: dict) -> dict:
    """Atomize a GenEval2 prompt into structured atoms with taxonomy context.

    This is the main entry point for Step 0. It takes a raw GenEval2 prompt
    record and produces the full atomized structure consumed by downstream
    T2I evaluation steps.

    Args:
        prompt_data: A GenEval2 record dict with keys:
            - "prompt" (str): The text prompt.
            - "vqa_list" (list): List of [question, answer] pairs.
            - "skills" (list, optional): Parallel list of skill labels.
            - "atom_count" (int, optional): Original atom count.
            - "prompt_id" (str, optional): Prompt ID from JSONL index.

    Returns:
        A dict with the following structure:
        {
            "prompt_id": "...",
            "prompt": "four brown monkeys and a metal bicycle",
            "atom_count": 6,
            "atoms": [...],
            "objects": [
                {
                    "object_name": "monkey",
                    "class_id": 376,
                    "class_name": "guenon",
                    "taxonomy_description": "...",
                    "diagnostic_checkpoints": {...},
                    "attributes_from_prompt": ["brown"],
                    "count_from_prompt": 4
                },
                ...
            ]
        }
    """
    prompt = prompt_data.get("prompt", "")
    vqa_list = prompt_data.get("vqa_list", [])
    skills = prompt_data.get("skills", [])
    prompt_id = prompt_data.get("prompt_id", "")

    # Step 1: Normalize atoms from VQA format
    atoms = normalize_atoms(vqa_list, skills)

    # Step 2: Extract unique objects
    object_names = extract_objects_from_atoms(atoms)

    # Step 3: Build object-to-classid mapping and link taxonomy
    object_to_classid = build_object_to_classid_map()

    # Step 4: For each object, gather attributes and count from atoms
    object_infos: list[dict] = []
    for obj_name in object_names:
        # Link taxonomy (returns None if object not in mapping)
        taxonomy_result = link_taxonomy(obj_name, object_to_classid)

        # Extract attributes and count from atoms for this object
        attributes: list[str] = []
        count: int | None = None
        for atom in atoms:
            target = atom.get("target_object", "")
            if not target:
                continue
            # Match singular or plural form of the target object
            target_singular = _singularize(target)
            if target_singular != obj_name:
                continue

            skill = atom.get("skill", "")
            if skill == "attribute":
                attr = _extract_attribute_from_question(atom["question"])
                if attr and attr not in attributes:
                    attributes.append(attr)
            elif skill == "count":
                count = _extract_count_from_answer(atom["answer"])

        # Build object info dict
        if taxonomy_result:
            obj_info = dict(taxonomy_result)
            obj_info["is_generic"] = False
        else:
            obj_info = {
                "object_name": obj_name,
                "class_id": None,
                "is_generic": True,  # NEW: mark as generic for Step 0d taxonomy generation
                "class_name": "",
                "taxonomy_description": "",
                "diagnostic_checkpoints": {},
            }
        obj_info["attributes_from_prompt"] = attributes
        obj_info["count_from_prompt"] = count
        object_infos.append(obj_info)

    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "atom_count": len(atoms),
        "atoms": atoms,
        "objects": object_infos,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  JSON Parsing Utilities (mirrors step1_router.py / step2_judge.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_json_response(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-len("```")]
    return text.strip()


def parse_json_safely(raw_text: str) -> dict | None:
    if raw_text is None or raw_text.strip() == "":
        return None
    cleaned = clean_json_response(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Generic Taxonomy Generation (Step 0d)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_generic_taxonomy(
    client: OpenAI,
    object_name: str,
    prompt_text: str,
    api_retry: int = 0,
    temperature: float = 0.0,
    model_name: str = "",
) -> dict:
    """为泛类物体生成通用诊断特征。

    Calls the LLM to produce taxonomy_description and diagnostic_checkpoints
    for an object that has no ImageNet class_id mapping (is_generic=True).
    Returns a dict matching the structure produced by link_taxonomy, with
    is_generic=True and class_name suffixed with "(generic)".
    """
    prompt = (
        f'你是生物分类学专家。给定物体名称 "{object_name}"'
        f'（来自 prompt: "{prompt_text}"），'
        '生成简洁但能区分该大类与其他大类的诊断特征。\n\n'
        '要求：\n'
        '1. 3-5 个 checkpoint，每个一句话\n'
        '2. 每个 checkpoint 必须是图片中可视觉验证的特征\n'
        '3. 不要涉及物种级特征（如特定花纹、颊囊等）\n'
        '4. 重点：能将该大类与其他易混淆大类区分开的特征\n'
        '5. 输出 JSON：{"taxonomy_description": "一句话总述", '
        '"diagnostic_checkpoints": {"部位": "特征描述"}}'
    )

    messages = [
        {"role": "system", "content": "You are a biology taxonomy expert. Output JSON only."},
        {"role": "user", "content": prompt},
    ]

    if not model_name:
        model_name = ATOMIZE_MODEL

    try:
        completion = api_call_with_retry(
            client.chat.completions.create,
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_retries=api_retry,
            label="GenericTaxonomy",
            extra_body={"enable_thinking": False},
        )
        raw = completion.choices[0].message.content
        if raw is None or raw.strip() == "":
            reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
            if reasoning:
                raw = reasoning
            else:
                return {
                    "object_name": object_name,
                    "is_generic": True,
                    "class_id": None,
                    "class_name": f"{object_name} (generic)",
                    "taxonomy_description": "",
                    "diagnostic_checkpoints": {},
                }
        result = parse_json_safely(raw)
        if result is None:
            return {
                "object_name": object_name,
                "is_generic": True,
                "class_id": None,
                "class_name": f"{object_name} (generic)",
                "taxonomy_description": "",
                "diagnostic_checkpoints": {},
            }
        return {
            "object_name": object_name,
            "is_generic": True,
            "class_id": None,
            "class_name": f"{object_name} (generic)",
            "taxonomy_description": result.get("taxonomy_description", ""),
            "diagnostic_checkpoints": result.get("diagnostic_checkpoints", {}),
        }
    except Exception as e:
        print(f"  [WARN] Generic taxonomy generation failed for '{object_name}': {e}")
        return {
            "object_name": object_name,
            "is_generic": True,
            "class_id": None,
            "class_name": f"{object_name} (generic)",
            "taxonomy_description": "",
            "diagnostic_checkpoints": {},
        }


def enrich_with_generic_taxonomy(
    atomized_data: dict,
    client: OpenAI,
    api_retry: int = 0,
    temperature: float = 0.0,
    model_name: str = "",
) -> dict:
    """为 atomized_data 中没有 taxonomy 的泛类物体生成通用诊断特征。

    原地修改 atomized_data["objects"]，为每个 is_generic=True 的物体
    调用 generate_generic_taxonomy 补全 taxonomy 信息。
    """
    objects = atomized_data.get("objects", [])
    for obj in objects:
        if obj.get("is_generic", False) and not obj.get("diagnostic_checkpoints"):
            print(f"  [Step 0d] Generating generic taxonomy for '{obj['object_name']}'...")
            generated = generate_generic_taxonomy(
                client, obj["object_name"],
                atomized_data.get("prompt", ""),
                api_retry=api_retry,
                temperature=temperature,
                model_name=model_name,
            )
            obj.update(generated)
    return atomized_data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Save / Load Atomized Prompts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_atomized_prompt(atomized: dict, output_dir: str | Path, prompt_id: str) -> str:
    """Save an atomized prompt structure to a JSON file.

    Args:
        atomized: The output of atomize_prompt().
        output_dir: Directory to save the file in.
        prompt_id: Prompt ID used to generate the filename.

    Returns:
        The absolute path to the saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"atomized_{prompt_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(atomized, f, indent=4, ensure_ascii=False)
    return str(filepath)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """CLI entry point: atomize a single GenEval2 prompt and save the result."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Atomize a GenEval2 prompt into structured atoms with taxonomy context."
    )
    parser.add_argument(
        "--jsonl", type=str, default=str(PROJECT_ROOT / "geneval2_data.jsonl"),
        help="Path to geneval2_data.jsonl (default: PROJECT_ROOT/geneval2_data.jsonl).",
    )
    parser.add_argument(
        "--prompt-id", type=str, required=True,
        help="0-based line index of the prompt in the JSONL file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(T2I_DIR / "output" / "atomized"),
        help="Directory to save the atomized JSON (default: t2i_harness/output/atomized).",
    )
    args = parser.parse_args()

    prompt_data = load_geneval2_prompt(args.jsonl, args.prompt_id)
    if prompt_data is None:
        print(f"[ERROR] Could not load prompt_id={args.prompt_id} from {args.jsonl}")
        return

    print(f"[INFO] Atomizing prompt: \"{prompt_data.get('prompt', '')}\"")

    atomized = atomize_prompt(prompt_data)

    filepath = save_atomized_prompt(atomized, args.output_dir, args.prompt_id)
    print(f"[INFO] Saved atomized prompt to: {filepath}")
    print(f"  - Atoms: {atomized['atom_count']}")
    print(f"  - Objects: {len(atomized['objects'])}")
    for obj in atomized["objects"]:
        class_info = f"class_id={obj['class_id']}" if obj["class_id"] else "no taxonomy"
        attrs = ", ".join(obj["attributes_from_prompt"]) or "none"
        print(f"    * {obj['object_name']} ({class_info}, count={obj['count_from_prompt']}, attrs=[{attrs}])")


if __name__ == "__main__":
    main()
