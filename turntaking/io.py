import json
import xml.etree.ElementTree as ET


def read_text(filename):
    with open(filename) as f:
        return f.read()


def read_json(json_file):
    with open(json_file) as f:
        return json.load(f)


def write_json(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f)


def load_xml(xml_file):
    return ET.parse(xml_file).getroot()
