import torch
import os
import csv
import re


class TwitchEmoteDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_file, emote_file, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer

        emote_list = TwitchEmoteDataset._read_dataset(emote_file)
        self.label_map = TwitchEmoteDataset._convert_emotes_to_label_ids(emote_list)
        self.emote_regex_dict = self._create_emote_regex_set(emote_list)

        raw_chat_string_list = TwitchEmoteDataset._read_dataset(dataset_file)
        self.dataset = self._create_dataset(raw_chat_string_list)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _convert_emotes_to_label_ids(emotes):
        emote_to_label_idx = {}
        for emote, i in enumerate(emotes):
            emote_to_label_idx[emote] = i
        return emote_to_label_idx

    @staticmethod
    def _read_dataset(file_path):
        """
        :param file_path: a string with the path to the .csv file dataset
        :return: a list of strings representing individual text chat instances
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError
        strings = []
        reader = csv.DictReader(file_path)
        for row in reader:
            strings.append(row['text'])
        return strings

    @staticmethod
    def _create_emote_regex_set(emote_set):
        """
        :param emote_set: a set of emote strings which will be our ultimate classification target
        :return: a corresponding dict of regex objects used to easily strip the text strings in preprocessing
        """
        emote_regex_dict = {}
        for emote in emote_set:
            emote_regex_dict[emote] = re.compile("(\s*){}(\s*)".format(emote))
        return emote_regex_dict

    def _get_emote_from_string(self, string):
        """
        :param string: input string
        :return: a tuple (string, emote_ids).
        string is the new string that is stripped of emotes,
        emote_ids is a list of ids representing the emotes that occur in the string. They can repeated to show
        frequency in the string
        """
        string = string.strip()
        emote_ids = []
        for emote in self.emote_regex_dict:
            string, num_subs_made = self.emote_regex_dict[emote].subn(string, '')
            emote_ids += [self.label_map[emote]] * num_subs_made
        return string, emote_ids

    def _hot_encode_emote_list(self, emote_ids):
        """
        :param emotes: list of emote indices
        :return: a hot encoded tensor representation
        """
        ret = torch.zeros(len(self.emote_regex_dict))
        for emote in emotes_ids:
            ret[emote] += 1
        return ret

    def _create_dataset(self, strings):
        """
        :param strings: a list of strings
        :return: a list with tuples (encoded_text_tensor, hot_encoded_emote_ids_tensor) as elements
        """
        dataset = []
        for string in strings:
            stripped_string, emote_ids = self._get_emote_from_string(string)
            input_tensor = self.tokenizer(stripped_string)
            emote_tensor = self._hot_encode_emote_list(emote_ids)
            dataset = (input_tensor, emote_tensor)
        return dataset
