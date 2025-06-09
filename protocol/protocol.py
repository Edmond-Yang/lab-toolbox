import os
import sys
os.environ.setdefault("PREPROCESSED_DIR", "/mnt/disk1")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Module imports
import yaml
import argparse

from enum import Enum
from utils.path import *
from utils.logger import *
from typing import List, Tuple, Optional, Union, Dict
from concurrent.futures import ThreadPoolExecutor


# Type
class CommandType(Enum):
    Contains = "+"
    NotContains = "-"

class Command:
    def __init__(self, command: Union[str, CommandType], names: List[str]) -> None:
        self.command = command
        self.names = names

        if type(command) is str:
            self.command = CommandType(command)

        if self.command not in CommandType:
            raise ValueError(f"Unsupported command: {self.command}")

    def _check(self, filename: Union[str, Path], name):
        if self.command == CommandType.Contains:
            return name in filename
        elif self.command == CommandType.NotContains:
            return name not in filename

    def __call__(self, x):
        if self.command == CommandType.Contains:
            return any([self._check(x, i) for i in self.names])
        elif self.command == CommandType.NotContains:
            return all([self._check(x, i) for i in self.names])

def generate_command(data: List[Dict[str, Union[str, List[str]]]]) -> List[Command]:
    return [Command(d['type'], d['text']) for d in data]

class Filter:
    def __call__(self, commands: List[Command], filename: Union[Path, str]) -> bool:
        return all([cmd(str(filename)) for cmd in commands])


class ProtocolGenerator:
    def _parse_subject(self, data):

        subject_prefix = data.get('subject_prefix', '')
        subject_pattern_prefix_zero = int(data.get('subject_pattern_prefix_zero', '0'))
        subject_pattern = data.get('subject_pattern', '')
        subject_postfix = data.get('subject_postfix', '')

        if subject_pattern == '':
            return []

        command_type = subject_pattern[0]
        subject_pattern = subject_pattern[1:]

        _subject = list()
        print(subject_pattern_prefix_zero)

        for part in subject_pattern.split(','):
            if '-' in part:
                start, end = part.split('-')
                start = int(start)
                end = int(end) + 1
                _subject.extend([f"{subject_prefix}{i:0{subject_pattern_prefix_zero}d}{subject_postfix}" for i in range(start, end)])
            else:
                _subject.append(f"{subject_prefix}{int(part):0{subject_pattern_prefix_zero}d}{subject_postfix}")

        return [{'type': command_type, 'text': _subject}]

    # General Example
    def _parse_videos(self, name: str, filter: List[Command]) -> List[Path]:

        F = Filter()
        # Get the video path
        video_path = pathManager.get_preprocessed_video_path(name, [], 'RGB')
        videos = list(video_path.glob('*'))

        # Filter the videos
        filtered_videos = [(name, v.stem) for v in videos if v.is_dir() and F(filter, v)]

        return filtered_videos

    # Specific Example
    def _parse_videos_from_VIPL(self, name: str, filter: List[Command]) -> List[Path]:
        F = Filter()
        # Get the video path
        video_path = pathManager.get_preprocessed_video_path(name, [], 'RGB')
        videos = list(video_path.glob('*/*'))

        # Filter the videos
        filtered_videos = [(name, v.parent.stem, v.stem) for v in videos if v.is_dir() and F(filter, v)]

        return filtered_videos

    def _get_all_matched_videos(self, datasets) -> List[Tuple[str, str]]:

        _matched_videos = list()

        for d in datasets:

            name = d.get('name')
            filter = d.get('filters', [])

            # subject
            subject = self._parse_subject(d)
            filter.extend(subject)

            print(f"Dataset: {name}, Filter: {filter}")

            # generate command
            commands = generate_command(filter)

            # parse videos
            if name == 'VIPL':
                _matched_videos.extend(self._parse_videos_from_VIPL(name, commands))
            else:
                _matched_videos.extend(self._parse_videos(name, commands))

        return _matched_videos

    def __call__(self, yaml_file) -> None:

        with open(yaml_file) as f:
            protocols = yaml.safe_load(f)

        for p in protocols.get('protocols', []):

            name = p['name']
            datasets = p['datasets']
            Logger.info(f"Protocol: {name}")

            files = self._get_all_matched_videos(datasets)
            Logger.info(f"Protocol {name}: Found {len(files)} videos")

            # Write the protocol
            protocol_file, _ = pathManager.get_protocol_path(name)
            with open(protocol_file, 'w') as f:
                for dataset_name, *video_names in files:
                    f.write(f'{dataset_name},{",".join(video_names)}\n')



if __name__ == '__main__':

    Logger.set_prefix('ProtocolGenerator')


    parser = argparse.ArgumentParser(description='Universal Protocol Generator')
    parser.add_argument('--template', type=str, default='./template.yaml', help='Template YAML file')
    args = parser.parse_args()

    pathManager.make_protocol_directory()
    generator = ProtocolGenerator()

    generator(args.template)