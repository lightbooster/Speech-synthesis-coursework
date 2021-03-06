import inflect
from googletrans import Translator


class MyTranslator:
    def __init__(self):
        # Some code
        self.translator = Translator()
        # Some code

    def translate(self, in_text, in_lang="en", out_lang="ru"):
        while True:
            try:
                result_ = self.translator.translate(in_text, src=in_lang, dest=out_lang)
                return result_
            except Exception as e:
                self.translator = Translator()
                print(e, "<---- Ошибка")


class RussianPreprocessor:
    """
    Preprocess of Cyrillic text for speech synthesis
    """

    __SYMBOLS = "-!'(),.:;? абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

    def __init__(self):
        self._symbols_dict = {self.__SYMBOLS[i]: i for i in range(len(self.__SYMBOLS))}
        self._my_translator = MyTranslator()
        self._inflect_engine = inflect.engine()

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

    def number_to_text(self, number: int, in_lang_="en", out_lang_="ru") -> str:
        string_num = self._inflect_engine.number_to_words(number)
        result = self._my_translator.translate(in_text=string_num, in_lang=in_lang_, out_lang=out_lang_)
        return result.text


def main() -> None:
    my_rp = RussianPreprocessor()
    # print(my_rp.number_to_text(151))


if __name__ == '__main__':
    main()
