
class RussianPreprocessor:
    """
    Preprocess of Cyrillic text for speech synthesis
    """

    __SYMBOLS = "-!'(),.:;? абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

    def __init__(self):
        self._symbols_dict = {self.__SYMBOLS[i]: i for i in range(len(self.__SYMBOLS))}

    def _preprocess_text(self, data: str):
        """
        Cleans Cyrillic
        :param data (str): source string
        :return (str): lowercased string, deleted repeated spaces, only Cyrillic alphabet and permitted symbols
        """
        preprocessed_data = ""
        data = data.strip().lower()
        is_prev_space = False
        for char in data:
            if char not in self.__SYMBOLS:
                continue

            if is_prev_space and char == ' ':
                continue

            if char == ' ':
                is_prev_space = True
            elif is_prev_space:
                is_prev_space = False

            preprocessed_data += char

        return preprocessed_data

    def text_to_sequence(self, data: str):
        """
        transform Cyrillic string into numeric sequence
        :param data:  Cyrillic string
        :return: (list) of ints
        """
        preprocessed_data = self._preprocess_text(data)
        encoded_chars = [self._symbols_dict[char] for char in preprocessed_data]
        return encoded_chars

