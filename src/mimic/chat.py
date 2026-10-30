"""Turns a typed sentence into a robot intent. Rule-based (no network, no API key): it understands the common ways
of saying go to / lock on an object, stop, pause, reset, how close to get, and a few questions."""
import re

# words that carry no object name ("go to the white cup" -> "white cup")
FILLER = set("""go goto move moves lock locked track follow approach find centre center pick reach get grab focus aim target
head navigate come look locate touch near closer toward towards to at on onto in into up the a an that this my our
please can could you would now over there and then it for me robot arm gripper hey just only lets let's us go-to
lock-in out so i want need""".split())

SYNONYMS = {"mug": "cup", "phone": "cell phone", "mobile": "cell phone", "glass": "wine glass", "plant": "potted plant",
            "computer": "laptop", "notebook": "laptop", "human": "person", "man": "person", "woman": "person",
            "people": "person", "monitor": "tv", "screen": "tv", "sofa": "couch", "bag": "backpack"}

DISTANCE = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|centimet\w*|mm|millimet\w*)")


def normalise(name):
    """'cups' -> 'cup', 'mug' -> 'cup' ..."""
    name = name.strip()
    if name in SYNONYMS:
        return SYNONYMS[name]
    if name.endswith("s") and name[:-1] in SYNONYMS | {"cup": 0, "bottle": 0, "laptop": 0, "book": 0, "person": 0}:
        return SYNONYMS.get(name[:-1], name[:-1])
    return name


def parse(text):
    """-> {"intent": ..., ...}. Intents: target(name[, stop_cm]) stop pause resume unlock reset yes no
    stop_distance(cm) look status help empty unknown"""
    t = re.sub(r"[?!.,]+", " ", text.lower()).strip()
    if not t:
        return {"intent": "empty"}
    dist = DISTANCE.search(t)
    cm = None
    if dist:
        cm = float(dist.group(1)) / (10.0 if dist.group(2).startswith(("mm", "milli")) else 1.0)
    rest = DISTANCE.sub(" ", t)
    rest = re.sub(r"\b(within|under|less than|away|from it|from the object|stay|keep|distance of|distance|stop at|get to|"
                  r"be)\b", " ", rest) if cm is not None else rest

    m = re.search(r"\b(?:fingertips?|finger tips?|gripper tips?|fingers|gripper)\b.*?(\d+(?:\.\d+)?)\s*(cm|mm)\s*"
                  r"(?:in front|ahead|beyond|past|out)", t)
    if m:
        return {"intent": "set", "key": "tip_offset_mm", "value": float(m.group(1)) * (10.0 if m.group(2) == "cm" else 1.0)}
    m = re.search(r"\b(?:cup|object|it|bottle|mug)\b.*?\b(?:is|are)\s*(\d+(?:\.\d+)?)\s*(cm|mm)\s*(?:tall|high)", t)
    if m:
        return {"intent": "set", "key": "object_height_mm", "value": float(m.group(1)) * (10.0 if m.group(2) == "cm" else 1.0)}
    m = re.search(r"\bcalibrate\b.*?(\d+(?:\.\d+)?)\s*(cm|mm)|\b(?:is|are)\s*(\d+(?:\.\d+)?)\s*(cm|mm)\s*(?:away|from the camera)", t)
    if m:
        v = float(m.group(1) or m.group(3))
        unit = m.group(2) or m.group(4)
        return {"intent": "calibrate", "cm": v / (10.0 if unit == "mm" else 1.0)}
    if t in ("yes", "y", "yep", "yeah", "confirm", "ok", "okay", "do it", "yes please", "go ahead", "go", "execute",
             "proceed", "run it", "execute the plan", "run the plan"):
        return {"intent": "yes"}
    if t in ("no", "n", "cancel", "never mind", "nevermind", "don't", "dont"):
        return {"intent": "no"}
    if re.match(r"^(stop|halt|freeze|abort|emergency|e stop|estop|e-stop|enough)\b", t) or (
            re.search(r"\b(stop|halt|freeze|abort|emergency|estop|e-stop|enough)\b", t) and cm is None
            and not re.search(r"\b(follow|track|lock|go|approach|find)\b", t)):
        return {"intent": "stop"}
    g = re.search(r"\b(open|close|shut|release|grab)\b.*\bgripper\b|\bgripper\b.*\b(open|close|shut)\b", t)
    if g:
        return {"intent": "gripper", "open": bool(re.search(r"\b(open|release)\b", t))}
    if re.search(r"\b(reset|home position|go home|default position|back to default)\b", t):
        return {"intent": "reset"}
    if re.search(r"\b(unlock|forget|release|drop (it|the target)|clear target)\b", t):
        return {"intent": "unlock"}
    if re.search(r"\b(pause|hold on|hold still|hold position|wait|stay (still|put))\b", t):
        return {"intent": "pause"}
    if re.search(r"^(resume|continue|carry on|go on|keep going|start)$", t):
        return {"intent": "resume"}
    if re.search(r"\b(what (do )?you see|what can you see|what is (there|visible)|list objects|what s (there|visible)|"
                 r"what do you detect|detect)\b", t):
        return {"intent": "look"}
    if re.search(r"\b(status|where are you|how far|what is the distance|how close|what are you doing)\b", t):
        return {"intent": "status"}
    if re.search(r"^(help|\?|commands|what can you do)$", t):
        return {"intent": "help"}

    words = [w for w in re.findall(r"[a-z][a-z\-]*", rest) if w not in FILLER]
    name = normalise(" ".join(words))
    if name and len(words) <= 4:
        out = {"intent": "target", "name": name}
        if cm is not None:
            out["stop_cm"] = cm
        return out
    if cm is not None:
        return {"intent": "stop_distance", "cm": cm}
    return {"intent": "unknown"}


HELP = ("Tell me what to go to: \"go to the cup\", \"lock on the bottle\", \"approach the white cup\", "
        "\"get within 5 cm of the cup\". Also: stop, pause, resume, unlock, reset, \"what do you see\", status. "
        "\"open the gripper\", \"fingertips are 13 cm in front of the camera\", \"the cup is 9 cm tall\", "
        "\"calibrate 30 cm\". I show the plan before moving. STOP always works and the arm only moves while I'm engaged.")
