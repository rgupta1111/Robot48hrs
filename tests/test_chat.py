from mimic.chat import parse


def intent(text):
    return parse(text)["intent"]


def test_targets():
    assert parse("go to the cup") == {"intent": "target", "name": "cup"}
    assert parse("Lock in on the white cup!") == {"intent": "target", "name": "white cup"}
    assert parse("please approach the bottle") == {"intent": "target", "name": "bottle"}
    assert parse("can you find my phone") == {"intent": "target", "name": "cell phone"}
    assert parse("follow the mug") == {"intent": "target", "name": "cup"}
    assert parse("cups") == {"intent": "target", "name": "cup"}


def test_target_with_distance():
    assert parse("go to the cup and stay 5 cm away") == {"intent": "target", "name": "cup", "stop_cm": 5.0}
    assert parse("approach the bottle within 80 mm") == {"intent": "target", "name": "bottle", "stop_cm": 8.0}


def test_distance_only():
    assert parse("get within 15 cm") == {"intent": "stop_distance", "cm": 15.0}


def test_controls():
    assert intent("stop") == "stop"
    assert intent("STOP!!") == "stop"
    assert intent("emergency stop") == "stop"
    assert intent("stop following the cup") == "stop"
    assert intent("please stop") == "stop"
    assert intent("pause") == "pause"
    assert intent("resume") == "resume"
    assert intent("unlock") == "unlock"
    assert intent("reset the arm to the home position") == "reset"
    assert intent("yes") == "yes" and intent("cancel") == "no"
    assert intent("what do you see?") == "look"
    assert intent("how far is it") == "status"
    assert intent("help") == "help"


def test_empty_and_unknown():
    assert intent("   ") == "empty"
    assert intent("go") == "unknown"
    assert intent("the") == "unknown"


def test_long_sentences_are_not_objects():
    assert intent("how about you tell me a really long story about robots") == "unknown"
