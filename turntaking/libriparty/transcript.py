from pathlib import Path
from turntaking.io import read_json
from turntaking.transcript import Transcript, ScenarioUtterance


def json_to_transcript(json):
    parsed_utts = []

    for spk, utts in json.items():
        if spk in ['noises', 'background']:
            continue

        for utt in utts:
            parsed_utts.append(ScenarioUtterance(
               speaker=spk,
               duration=utt['stop'] - utt['start'],
               text=utt['words'],
               start_time=utt['start']
            ))

    return Transcript(parsed_utts)


def yield_libriparty_transcript(data):
    json_file = read_json(data / 'metadata' / 'train.json')
    for session, speakers in json_file.items():
        transcript = json_to_transcript(speakers)
        yield session, transcript
