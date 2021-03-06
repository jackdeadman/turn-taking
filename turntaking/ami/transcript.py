from pathlib import Path

from tqdm import tqdm

from turntaking.extra_typing import PathString
from turntaking.io import load_xml
from turntaking.transcript import ScenarioUtterance, Transcript
import xml

from turntaking.utils import num_to_letter


def yield_ami_files(pattern):
    def get_file(i):
        p = Path(pattern.format(num_to_letter(i, upper=True)))
        return p if p.exists() else None

    i = 0
    file = get_file(i)
    while file:
        yield file
        i += 1
        file = get_file(i)


def load_segments(pattern):
    """
    :param pattern: path/to/EN2001a.{}.segments.xml
    :return:
    """
    xml_files = [(f, load_xml(f)) for f in yield_ami_files(pattern)]

    for file, xml_file in xml_files:
        for segment in xml_file:
            dict = segment.attrib
            yield {
                'speaker': str(file),
                'start_time': float(dict['transcriber_start']),
                'end_time': float(dict['transcriber_end']),
            }


def load_ami_transcript(session, base: PathString='data/ami/segments'):
    """
    Currently just loading the segments file. So no actual text is loaded.
    :param path: currently assumes a path to a *CHiME-5* transcript
    :return: loaded transcript
    """
    segments = load_segments(str(Path(base) / (session + '.{}.segments.xml')))

    return Transcript([
        ScenarioUtterance(
            speaker=ami_segment['speaker'],
            text="",
            start_time=ami_segment['start_time'],
            duration=ami_segment['end_time'] - ami_segment['start_time']
        )
        for ami_segment in segments
    ])


def yield_ami_transcripts(base, num_spks):
    sessions = get_ami_sessions_with_same_number_of_speakers(num_spks)
    for session in sessions:
        transcript = load_ami_transcript(session, base=Path(base) / 'ami/segments')
        yield session, transcript


def get_ami_sessions():
    return ["EN2001a", "EN2001b", "EN2001d", "EN2001e", "EN2002a", "EN2002b",
            "EN2002c", "EN2002d", "EN2003a", "EN2004a", "EN2005a", "EN2006a",
            "EN2006b", "EN2009b", "EN2009c", "EN2009d", "ES2002a", "ES2002b",
            "ES2002c", "ES2002d", "ES2003a", "ES2003b", "ES2003c", "ES2003d",
            "ES2004a", "ES2004b", "ES2004c", "ES2004d", "ES2005a", "ES2005b",
            "ES2005c", "ES2005d", "ES2006a", "ES2006b", "ES2006c", "ES2006d",
            "ES2007a", "ES2007b", "ES2007c", "ES2007d", "ES2008a", "ES2008b",
            "ES2008c", "ES2008d", "ES2009a", "ES2009b", "ES2009c", "ES2009d",
            "ES2010a", "ES2010b", "ES2010c", "ES2010d", "ES2011a", "ES2011b",
            "ES2011c", "ES2011d", "ES2012a", "ES2012b", "ES2012c", "ES2012d",
            "ES2013a", "ES2013b", "ES2013c", "ES2013d", "ES2014a", "ES2014b",
            "ES2014c", "ES2014d", "ES2015a", "ES2015b", "ES2015c", "ES2015d",
            "ES2016a", "ES2016b", "ES2016c", "ES2016d", "IB4001.", "IB4002.",
            "IB4003.", "IB4004.", "IB4005.", "IB4010.", "IB4011.", "IN1001.",
            "IN1002.", "IN1005.", "IN1007.", "IN1008.", "IN1009.", "IN1012.",
            "IN1013.", "IN1014.", "IN1016.", "IS1000a", "IS1000b", "IS1000c",
            "IS1000d", "IS1001a", "IS1001b", "IS1001c", "IS1001d", "IS1002b",
            "IS1002c", "IS1002d", "IS1003a", "IS1003b", "IS1003c", "IS1003d",
            "IS1004a", "IS1004b", "IS1004c", "IS1004d", "IS1005a", "IS1005b",
            "IS1005c", "IS1006a", "IS1006b", "IS1006c", "IS1006d", "IS1007a",
            "IS1007b", "IS1007c", "IS1007d", "IS1008a", "IS1008b", "IS1008c",
            "IS1008d", "IS1009a", "IS1009b", "IS1009c", "IS1009d", "TS3003a",
            "TS3003b", "TS3003c", "TS3003d", "TS3004a", "TS3004b", "TS3004c",
            "TS3004d", "TS3005a", "TS3005b", "TS3005c", "TS3005d", "TS3006a",
            "TS3006b", "TS3006c", "TS3006d", "TS3007a", "TS3007b", "TS3007c",
            "TS3007d", "TS3008a", "TS3008b", "TS3008c", "TS3008d", "TS3009a",
            "TS3009b", "TS3009c", "TS3009d", "TS3010a", "TS3010b", "TS3010c",
            "TS3010d", "TS3011a", "TS3011b", "TS3011c", "TS3011d", "TS3012a",
            "TS3012b", "TS3012c", "TS3012d"]


def get_files_for_session(base, sess):
    pattern = str(Path(base) / (sess + '.{}.segments.xml'))
    return list(yield_ami_files(pattern))


def get_ami_sessions_with_same_number_of_speakers(num_spks, base='data/ami/segments'):
    sessions = get_ami_sessions()
    for sess in sessions:
        files = get_files_for_session(base, sess)
        if len(files) == num_spks:
            yield sess



def main():
    import matplotlib.pyplot as plt
    from turntaking.vis import plot_number_of_people_speaking

    base = '/media/jack/LaCie/interspeech2022/data/ami/segments/'

    output = Path('output/ami_plots/overlap')
    output.mkdir(exist_ok=True, parents=True)

    for sess in tqdm(list(
            get_ami_sessions_with_same_number_of_speakers(base, num_spks=4))):
        try:
            plt.figure()
            plt.title(sess)
            plt.ylim(0, 1)
            transcript = load_ami_transcript(base, session=sess)
            samples = transcript.samples(sample_rate=1000)
            if len(samples[0]) != 4:
                continue
            plot_number_of_people_speaking(samples)
            plt.savefig(str(output / (sess + '.png')))
        except:
            print('Failed: ', sess)

if __name__ == '__main__':
    main()

