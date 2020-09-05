"""Utilities for generating error messages.

Processed properties are a way of defining property attributes for a class that
require input sanitation, type checking, function application, error message
generation etc. from user-supplied values in a manner that minimises code
reuse.

"""

from typing import Iterable


def format_case(item, case):
    """Allow :obj:`str` case formatting method application from keyword.

    Parameters
    ----------
    item : str
        Item to be case formatted.
    case : str
        Which case format method to use.

    Returns
    -------
    str
        :arg:`item` with case method applied.
    """
    if case == "title":
        return item.title()
    elif case == "upper":
        return item.upper()
    elif case == "lower":
        return item.lower()
    else:
        return item


def format_for_output(items, *args, **kwargs):
    """Utility method for formatting console output.

    Passes directly to :func:`format_multiple_items_for_output` just with a
    shorter function name.

    Parameters
    ----------
    items : iterable
        Items to be formatted for output.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    str
        Formatted string for console output.
    """
    return format_multiple_items_for_output(items, *args, **kwargs)


def format_multiple_items_for_output(items, wrapping_char="'", *,
                                     prefix_char="", case=None,
                                     with_verb=False, with_or=False,
                                     with_preposition=False):
    """Format multiple items for pretty console output.

    Args
    ----
    items : iterable of str
        Items to be formatted.
    wrapping_char : str (default `"'"`)
        Prefix and suffix character for format wrapping.
    prefix_char : str (default `""`)
        Additional prefix.
    case : str (default `None`)
        Keyword for :func:`format_case`.
    with_verb : bool, optional (default `False`)
        Append the correct conjugation of "is"/"are" to end of list.

    Returns
    -------
    str
        Formatted string of multiple items for console output.
    """
    items = (items, ) if isinstance(items, str) else items
    items = [f"{prefix_char}{format_case(item, case)}" for item in items]
    if len(items) == 1:
        formatted_items = f"{wrapping_char}{items[0]}{wrapping_char}"
        if with_preposition and wrapping_char == "":
            first_word, _ = formatted_items.split(maxsplit=1)
            starts_with_vowel = first_word[0] in {"a", "e", "h", "i", "o", "u"}
            is_acronym = first_word.upper() == first_word
            if starts_with_vowel or is_acronym:
                preposition = "an"
            else:
                preposition = "a"
            formatted_items = " ".join([preposition, formatted_items])
    else:
        pad = f"{wrapping_char}, {wrapping_char}"
        joiner = "or" if with_or else "and"
        formatted_items = (f"{wrapping_char}{pad.join(items[:-1])}"
                           f"{wrapping_char} {joiner} {wrapping_char}"
                           f"{items[-1]}{wrapping_char}")
    verb = "is" if len(items) == 1 else "are"
    if with_verb:
        formatted_items = f"{formatted_items} {verb}"

    return formatted_items
