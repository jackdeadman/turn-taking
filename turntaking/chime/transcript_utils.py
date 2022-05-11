# https://github.com/chimechallenge/prepare-CHiME5-sample/blob/master/transcript_utils.py

import json
import datetime
# from phd_package.datasets import valid_utterances
# from phd_package.datasets import find_transcriptions_file


def valid_utterances(transcriptions):
    for utt in transcriptions:
        if 'speaker' in utt:
            yield utt


def time_text_to_float(time_string):
    """Convert tramscript time from text to float format."""
    hours, minutes, seconds = time_string.split(':')
    seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return seconds


def time_float_to_text(time_float):
    """Convert tramscript time from float to text format."""
    # Milliseconds are rounded to 2 dp.
    time = datetime.datetime.min + datetime.timedelta(seconds=time_float+0.005)
    return time.strftime('%H:%M:%S.%f')[:-4]


def load_transcript(file, convert=False, chime6=False):
    """Load final merged transcripts.
    session: recording session name, e.g. 'S12'
    """
    with open(file) as f:
        transcript = json.load(f)
    if convert:
        if chime6:
            for item in transcript:
                item['start_time'] = time_text_to_float(item['start_time'])
                item['end_time'] = time_text_to_float(item['end_time'])
        else:
            for item in transcript:
                for key in item['start_time']:
                    item['start_time'][key] = time_text_to_float(item['start_time'][key])
                for key in item['end_time']:
                    item['end_time'][key] = time_text_to_float(item['end_time'][key])
    return transcript


def save_transcript(transcript, session, root, convert=False, challenge='CHIME5'):
    """Save transcripts to json file."""

    # Need to make a deep copy so time to string conversions only happen locally
    transcript_copy = [element.copy() for element in transcript]

    if convert:
        for item in transcript_copy:
            # For CHiME-5 every time is a dictionary with entries for each device
            if type(item['start_time']) == dict:
                for key in item['start_time']:
                    item['start_time'][key] = time_float_to_text(
                        item['start_time'][key])
                for key in item['end_time']:
                    item['end_time'][key] = time_float_to_text(
                        item['end_time'][key])
            else:
                # For CHiME-6 there is only one start and end time per utterance.
                item['start_time'] = time_float_to_text(item['start_time'])
                item['end_time'] = time_float_to_text(item['end_time'])

    with open(root + "/" + session + ".json", 'w') as outfile:
        json.dump(transcript_copy, outfile, indent=4)


def compute_utt_id(utt, chime6):

    def fix(time):
        return int(time * 100)

    if chime6:
        return '{}_{}-{:07d}-{:07d}'.format(
                       utt['speaker'], utt['session_id'],
                       fix(utt['start_time']), fix(utt['end_time']))
    else:
        return '{}_{}-{:07d}-{:07d}'.format(utt['speaker'], utt['session_id'], round(utt['start_time']['original']), round(utt['end_time']['original'])).replace('.', '')


def find_competing_speakers(file, chime6=False):
    """
    Finds all the utterances in a transcript
    :param chime6:
    :param file:
    :return: list of utterances with no competing speakers
    """
    transcript = load_transcript(file, convert=True, chime6=chime6)
    active = list()
    competing_speakers = {}

    def remove_finished_utt(time):
        if chime6:
            return [utt for utt in active if utt['end_time'] > time]
        else:
            return [utt for utt in active if utt['end_time']['original'] > time]

    # Skip ones that are redacted
    for utt in valid_utterances(transcript):
        start_time = utt['start_time'] if chime6 else utt['start_time']['original']

        utt_id = compute_utt_id(utt, chime6)
        active = remove_finished_utt(start_time)
        active.append(utt)
        competing_speakers[utt_id] = active[::]

    return competing_speakers


def find_utterances_with_n_competing_speakers(file, n, chime6=False, return_utts=False):
    competing_speakers = find_competing_speakers(file, chime6)
    if return_utts:
        return {utt: speakers for utt, speakers in competing_speakers.items() if len(speakers) == n}
    return set({utt for utt, speakers in competing_speakers.items() if len(speakers) == n})


def find_isolated_utterances(file, chime6=False, return_utts=False):
    return find_utterances_with_n_competing_speakers(file, n=1, chime6=chime6, return_utts=return_utts)

