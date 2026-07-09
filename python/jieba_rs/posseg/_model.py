import json
import os


def decode_state(value):
    return value[0], value[1:]


path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hmm_data.json")
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)["posseg"]

start_P = {decode_state(key): value for key, value in data["start"].items()}
trans_P = {
    decode_state(key): {decode_state(target): value for target, value in targets.items()}
    for key, targets in data["trans"].items()
}
emit_P = {
    decode_state(key): values
    for key, values in data["emit"].items()
}
char_state_tab_P = {
    char: tuple(decode_state(value) for value in values)
    for char, values in data["char_state_tab"].items()
}
