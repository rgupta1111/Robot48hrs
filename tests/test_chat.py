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
    assert intent("to the") == "unknown"
    assert intent("the") == "unknown"


def test_long_sentences_are_not_objects():
    assert intent("how about you tell me a really long story about robots") == "unknown"


def test_gripper():
    assert parse("open the gripper") == {"intent": "gripper", "open": True}
    assert parse("close the gripper please") == {"intent": "gripper", "open": False}


def test_settings_and_calibration():
    assert parse("fingertips are 13 cm in front of the camera") == {"intent": "set", "key": "tip_offset_mm", "value": 130.0}
    assert parse("the cup is 9 cm tall") == {"intent": "set", "key": "object_height_mm", "value": 90.0}
    assert parse("calibrate 30 cm") == {"intent": "calibrate", "cm": 30.0}
    assert parse("the cup is 25 cm away") == {"intent": "calibrate", "cm": 25.0}
    assert intent("go") == "yes" and intent("execute the plan") == "yes"


def test_pick_and_handover():
    assert parse("pick up the cup") == {"intent": "pick", "name": "cup", "then": None}
    assert parse("grab the bottle") == {"intent": "pick", "name": "bottle", "then": None}
    assert parse("pick up the cup and bring it to the closest person") == {"intent": "pick", "name": "cup", "then": "person"}
    assert parse("pick the cup up then give it to the nearest person") == {"intent": "pick", "name": "cup", "then": "person"}
    assert parse("bring it to the closest person") == {"intent": "person"}
    assert parse("hand it over to someone") == {"intent": "person"}
    assert parse("go to the cup") == {"intent": "target", "name": "cup"}          # plain reaching is unchanged
    assert parse("open the gripper") == {"intent": "gripper", "open": True}


def test_object_width_from_above():
    assert parse("the cup is 8 cm across") == {"intent": "set", "key": "object_top_mm", "value": 80.0}
    assert parse("the cup is 9 cm tall") == {"intent": "set", "key": "object_height_mm", "value": 90.0}


def test_choosing_between_several_cups():
    assert parse("switch to the other cup") == {"intent": "switch", "which": "other"}
    assert parse("the left one") == {"intent": "switch", "which": "left"}
    assert parse("use the closest cup") == {"intent": "switch", "which": "closest"}
    assert parse("cup 2") == {"intent": "switch", "which": 2}
    assert parse("the second cup") == {"intent": "switch", "which": 2}
    assert parse("go to the right cup") == {"intent": "target", "name": "cup", "which": "right"}
    assert parse("pick up the other cup") == {"intent": "pick", "name": "cup", "then": None, "which": "other"}
    assert parse("go to the cup") == {"intent": "target", "name": "cup"}          # unchanged without a qualifier
    assert parse("stop") == {"intent": "stop"}


def test_numbered_objects_other_than_cups():
    assert parse("person 3") == {"intent": "target", "name": "person", "which": 3}      # the app turns this into a switch when already working on a person
    assert parse("get within 5 cm of the cup") == {"intent": "target", "name": "cup", "stop_cm": 5.0}
