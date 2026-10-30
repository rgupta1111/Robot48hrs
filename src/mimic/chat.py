"""Turns a typed sentence into a robot intent. Rule-based (no network, no API key): it understands the common ways
of saying go to / lock on an object, stop, pause, reset, how close to get, and a few questions."""
import re

# words that carry no object name ("go to the white cup" -> "white cup")
FILLER = set("""go goto move moves lock locked track follow approach find centre center pick reach get grab focus aim target
head navigate come look locate touch near closer toward towards to at on onto in into up the a an that this my our
please can could you would now over there and then it for me robot arm gripper hey just only lets let's us go-to
lock-in out so i want need of from by with within than""".split())

SYNONYMS = {"mug": "cup", "phone": "cell phone", "mobile": "cell phone", "glass": "wine glass", "plant": "potted plant",
            "computer": "laptop", "notebook": "laptop", "human": "person", "man": "person", "woman": "person",
            "people": "person", "monitor": "tv", "screen": "tv", "sofa": "couch", "bag": "backpack"}

QUALIFIERS = {"left", "leftmost", "right", "rightmost", "closest", "nearest", "nearer", "farthest", "furthest", "other", "another",
              "next", "previous", "different", "first", "second", "third", "1st", "2nd", "3rd", "biggest", "largest", "smallest"}
SWITCH_WORDS = {"switch", "swap", "change", "toggle", "use", "choose", "select", "instead", "one", "ones"}
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3}


def which_one(t):
    """Which of several objects of the same kind is meant: 'other', 'prev', 'left', 'right', 'closest', 'farthest' or its
    1-based number; None if the sentence does not say."""
    m = re.search(r"(?:\b(?:cups?|objects?|bottles?|mugs?|persons?|people|laptops?|number|no|item)\s*#?|#)\s*(\d)\b", t)
    if m:
        return int(m.group(1))
    for w, n in ORDINALS.items():
        if re.search(rf"\b{w}\b", t):
            return n
    if re.search(r"\bprevious\b", t):
        return "prev"
    if re.search(r"\b(other|another|next|different)\b", t):
        return "other"
    if re.search(r"\b(left|leftmost)\b", t):
        return "left"
    if re.search(r"\b(right|rightmost)\b", t):
        return "right"
    if re.search(r"\b(closest|nearest|nearer|biggest|largest)\b", t):
        return "closest"
    if re.search(r"\b(farthest|furthest|smallest)\b", t):
        return "farthest"
    return None


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
    m = re.search(r"\b(?:cup|object|it|bottle|mug)\b.*?\b(?:is|are)\s*(\d+(?:\.\d+)?)\s*(cm|mm)\s*(?:across|wide|in diameter|diameter)", t)
    if m:
        return {"intent": "set", "key": "object_top_mm", "value": float(m.group(1)) * (10.0 if m.group(2) == "cm" else 1.0)}
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

    handover = re.search(r"\b(?:and |then )?(?:bring|give|hand|deliver|pass|carry|take)\b.*\b(?:person|human|someone|somebody|people|man|woman)\b", t)
    head = t[:handover.start()] if handover else t
    which = which_one(head)                                  # a qualifier about the OBJECT, not about the person
    if which is not None and not handover and not re.search(r"\b(stop|halt|reset|unlock|status|help)\b", t):
        go_verb = re.search(r"\b(go|move|approach|pick|grab|grasp|fetch|reach|take|bring|lift|follow|track|find)\b", t)
        keep = [w for w in re.findall(r"[a-z][a-z\-]*", t) if w not in FILLER | QUALIFIERS | SWITCH_WORDS | {"cup", "cups"} | set(ORDINALS)]
        if not go_verb and not keep:
            return {"intent": "switch", "which": which}
    pickv = re.search(r"\b(pick up|pick|grab|grasp|fetch|take|lift)\b", head)
    if pickv:
        pw = [w for w in re.findall(r"[a-z][a-z\-]*", head) if w not in FILLER | {"up", "pick", "grab", "grasp", "fetch", "take", "lift", "and", "then"}]
        pw = [w for w in pw if w not in QUALIFIERS and w not in SWITCH_WORDS]
        name = normalise(" ".join(pw))
        if name and len(pw) <= 4:
            out = {"intent": "pick", "name": name, "then": "person" if handover else None}
            if which is not None:
                out["which"] = which
            return out
    if handover:
        return {"intent": "person"}

    words = [w for w in re.findall(r"[a-z][a-z\-]*", rest) if w not in FILLER and w not in QUALIFIERS and w not in SWITCH_WORDS]
    name = normalise(" ".join(words))
    if name and len(words) <= 4:
        out = {"intent": "target", "name": name}
        if cm is not None:
            out["stop_cm"] = cm
        if which is not None:
            out["which"] = which
        return out
    if cm is not None:
        return {"intent": "stop_distance", "cm": cm}
    return {"intent": "unknown"}


HELP = ("Tell me what to go to: \"go to the cup\", \"lock on the bottle\", \"approach the white cup\", "
        "\"get within 5 cm of the cup\". Also: stop, pause, resume, unlock, reset, \"what do you see\", status. "
        "\"switch to the other cup\", \"the left cup\", \"cup 2\" (or click the cup on the video), \"pick up the cup\", \"pick up the cup and bring it to the closest person\", \"bring it to the closest person\", \"open the gripper\", \"fingertips are 13 cm in front of the camera\", \"the cup is 9 cm tall\", "
        "\"calibrate 30 cm\". I show the plan before moving. STOP always works and the arm only moves while I'm engaged.")
