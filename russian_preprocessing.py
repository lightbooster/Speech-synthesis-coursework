import inflect
from num_to_text import num2text
from googletrans import Translator


class MyTranslator:
    def __init__(self):
        # Some code
        self.translator = Translator()
        # Some code

    def translate(self, in_text, in_lang="en", out_lang="ru"):
        while True:
            try:
                result_ = self.translator.translate(in_text,
                                                    src=in_lang,
                                                    dest=out_lang)
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
        self._symbols_dict = {self.__SYMBOLS[i]: i for i in
                              range(len(self.__SYMBOLS))}
        self._my_translator = MyTranslator()
        # self._inflect_engine = inflect.engine()

    def _preprocess_text(self, data: str):
        """
        Cleans Cyrillic
        :param data (str): source string
        :return (str): lowercased string, deleted repeated spaces,
                       only Cyrillic alphabet and permitted symbols
        """
        preprocessed_data = ""
        data = data.strip().lower()
        is_prev_space = False

        num_str = ""
        is_num = False

        for char in data:
            if char.isdecimal():
                if not is_num and preprocessed_data[-1] != " ":
                    preprocessed_data += " "
                is_num = True
                num_str += char
                continue
            elif is_num:
                is_num = False
                preprocessed_data += num2text(int(num_str))
                num_str = ""

            if char not in self.__SYMBOLS:
                continue

            if is_prev_space and char == ' ':
                continue

            if char == ' ':
                is_prev_space = True
            elif is_prev_space:
                is_prev_space = False

            preprocessed_data += char

        if is_num:
            preprocessed_data += num2text(int(num_str))

        return preprocessed_data

    def text_to_sequence(self, data: str):
        """
        transform Cyrillic string into numeric sequence
        :param data:  Cyrillic string
        :return: (list) of ints
        """
        preprocessed_data = self._preprocess_text(data)
        encoded_chars = [self._symbols_dict[char] for char in
                         preprocessed_data]
        return encoded_chars

    def number_preprocessor(self, number: int, in_lang_="en",
                            out_lang_="ru") -> str:
        inflect.engine()
        string_num = inflect.engine().number_to_words(number)
        # print(string_num)
        result = self._my_translator.translate(in_text=string_num,
                                               in_lang=in_lang_,
                                               out_lang=out_lang_)
        return result.text


def main() -> None:
    my_rp = RussianPreprocessor()
    # print(my_rp.number_to_text(10))
    # print(num2text(50))
    # print(my_rp.number_preprocessor(50))
    print(my_rp._preprocess_text("с 10*50"))
    print(my_rp.text_to_sequence("с 10*50"))


if __name__ == '__main__':
    main()
