import os

import nltk.tokenize
import torch
from tqdm import tqdm


class TwitchEmoteDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_file, emote_file):
        super().__init__()

        self.word_tokenizer = nltk.TweetTokenizer()
        self.emotes = TwitchEmoteDataset._read_string_file(emote_file)
        self.emote_ids = TwitchEmoteDataset._convert_emotes_to_label_ids(self.emotes)

        self.dataset, self.stripped_chat_string_list = self._create_dataset(dataset_file)
        self.vocab_frequencies = self._get_vocab_and_freqs()

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1]

    def __len__(self):
        return len(self.dataset)

    def _get_vocab_and_freqs(self):
        vocab_dict = {}
        for words in self.stripped_chat_string_list:
            for word in words:
                vocab_dict[word] = vocab_dict.get(word, 0) + 1
        return vocab_dict

    @staticmethod
    def _convert_emotes_to_label_ids(emotes):
        emote_to_label_idx = {}
        for i, emote in enumerate(emotes):
            emote_to_label_idx[emote] = i + 1
        return emote_to_label_idx

    @staticmethod
    def _read_string_file(file_path):
        """
        :param file_path: a string with the path to the .csv file dataset
        :return: a list of strings representing individual text chat instances
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError
        strings = []
        f = open(file_path, 'r')
        reader = f.readlines()
        for row in reader:
            strings.append(row.strip().rstrip())
        f.close()
        return strings

    def _get_emote_from_string(self, string):
        """
        :param string: input string
        :return: a tuple (string, emote_ids).
        string is the new list of words that is stripped of emotes,
        emote_ids is a list of ids representing the emotes that occur in the string. They can repeated to show
        frequency in the string
        """

        emote_ids = []
        string = self.word_tokenizer.tokenize(string)
        ret = string.copy()
        for emote in self.emotes:
            count = ret.count(emote)
            emote_ids += [self.emote_ids[emote]] * count
            if count > 0:
                ret.remove(emote)
        if len(emote_ids) == 0:
            emote_ids = []
            ret = []
        return ret, emote_ids

    def _hot_encode_emote_list(self, emote_ids):
        """
        :param emote_ids: list of emote indices
        :return: a hot encoded tensor representation
        """
        ret = torch.zeros(len(self.emotes) + 1)
        for emote in emote_ids:
            ret[emote] += 1
        return ret

    def get_emote(self, emote_id):
        if int(emote_id) == 0:
            return "None"
        for key in self.emote_ids:
            val = self.emote_ids[key]
            if val == emote_id:
                return key

    def check_emote_counts(self):
        ret = torch.zeros(len(self.emotes) + 1)
        for i in self:
            ret += i[1]
        return ret

    def _create_dataset(self, dataset_file):
        """
        :param dataset_file: file with chat entries seperated by newline
        :return: a list with tuples (encoded_text_tensor, hot_encoded_emote_ids_tensor) as elements
        """
        strings = self._read_string_file(dataset_file)

        dataset, stripped_strings = [], []
        for string in tqdm(strings):
            stripped_string, emote_ids = self._get_emote_from_string(string)
            stripped_strings.append(stripped_string)
            emote_tensor = self._hot_encode_emote_list(emote_ids)
            if len(stripped_strings) == 0:
                continue
            dataset.append((stripped_string, emote_tensor))
        return dataset, stripped_strings
