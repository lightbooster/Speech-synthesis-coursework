import re
from num_to_text import num2text

__SYMBOLS = "-!'(),.:;? абвгдеёжзийклмнопрстуфхцчшщъыьэюя\n"

__ABBREVATIONS = (
    (r'т\.д\.', 'так далее'),
    (r'т\.е\.', 'то есть'),
    (r'т\.к\.', 'так как'),
    (r'г\.', 'год'),
    (r'гг\.', 'годы'),
    (r'г\.г\.', 'годы'),
    (r'н\.э\.', 'нашей эры')
)


def _filter_symbols(text: str, add_permitted="", replace_symbol='',
                    is_decimal_permitted=True) -> str:
    return ''.join([char if (char in __SYMBOLS
                             or char in add_permitted
                             or (is_decimal_permitted and char.isdecimal()))
                    else replace_symbol
                    for char in text])


def _collapse_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text)


def _replace_decimal_point(text: str, replace_str='целых') -> str:
    return re.sub(r'([0-9]+\.[0-9]+)',
                  lambda x: x.group(1).replace('.', ' ' + replace_str),
                  text)


def _replace_numbers(text: str) -> str:
    return re.sub(r'([0-9]+)',
                  lambda x: num2text(int(x.group(1))),
                  text)


def _replace_abbs(text: str) -> str:
    for abb in __ABBREVATIONS:
        text = re.sub(abb[0], abb[1], text)
    return text


def _encode_text(text: str) -> list:
    symbols_dict = {__SYMBOLS[i]: i for i in range(len(__SYMBOLS))}
    return [symbols_dict[char] for char in text]


def clear_text(text: str) -> str:
    text = text.lower()
    text = _filter_symbols(text, replace_symbol='')
    text = _collapse_spaces(text)
    text = _replace_decimal_point(text, replace_str='')
    text = _replace_numbers(text)
    text = _replace_abbs(text)
    return text


def text_to_sequence(text: str) -> list:
    text = clear_text(text)
    return _encode_text(text)

# test_text = "Сейчас 2021 г. н.э.. " \
#             "На уЛиЦе 28 градусов ТЕПЛА, т.е. очень * жарко*! " \
#             "Встретимся       в 20.30 у вокзала"

# print(clear_text(test_text))

# >>> сейчас две тысячи двадцать один год нашей эры.
# на улице двадцать восемь градусов тепла, то есть очень жарко!
# встретимся в двадцать тридцать у вокзала

# print(text_to_sequence(test_text))

# >>> [29, 16, 21, ... 11, 23, 11]
