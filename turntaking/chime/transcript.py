from turntaking.chime.transcript_utils import load_transcript
from turntaking.extra_typing import PathString
from turntaking.transcript import Transcript, ScenarioUtterance
from pathlib import Path


def load_chime_transcript(path: PathString):
    """
    IMPORTANT: if an utterance is redacted, it is simply ignored
    :param path: currently assumes a path to a *CHiME-5* transcript
    :return: loaded transcript
    """
    loaded_json = load_transcript(str(path), convert=True)

    return Transcript([
        ScenarioUtterance(
            speaker=chime_utterance['speaker'],
            text=chime_utterance['words'],
            start_time=chime_utterance['start_time']['original'],
            duration=chime_utterance['end_time']['original'] - chime_utterance['start_time']['original']
        )
        for chime_utterance in loaded_json if 'speaker' in chime_utterance
    ])


def yield_transcripts(data_dir: PathString):
    data_dir = Path(data_dir)
    for json_file in data_dir.glob('**/*.json'):
        session_name = json_file.name.split('.json')[0]
        yield session_name, load_chime_transcript(json_file)
