# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.te.graph_utils import (
    NEMO_CHAR,
    NEMO_TE_DIGIT,
    NEMO_WHITE_SPACE,
    GraphFst,
)
from nemo_text_processing.inverse_text_normalization.te.utils import get_abs_path


class TelephoneFst(GraphFst):
    """
    Classifies Telugu telephone, pincode, and four-digit credit sequences.

    Example:
        తొమ్మిది ఎనిమిది ఏడు ఆరు ఐదు నాలుగు మూడు రెండు ఒకటి సున్నా
        -> telephone { number_part: "౯౮౭౬౫౪౩౨౧౦" }

    Supports ten-digit mobile numbers with a nonzero first digit,
    eleven-digit landlines starting with zero, six-digit pincodes,
    and four-digit credit sequences. Context cues and up to five
    surrounding nonnumeric words are preserved. Country codes use
    two spoken digits.
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        nonzero = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        context_cues = pynini.string_file(get_abs_path("data/telephone/context_cues.tsv"))
        country_symbol = pynini.string_file(get_abs_path("data/telephone/country_symbol.tsv"))
        digit = (zero | nonzero).optimize()
        whitespace = pynini.closure(NEMO_WHITE_SPACE, 1)
        delete_separator = pynutil.delete(whitespace)
        retain_separator = pynini.cross(whitespace, " ")
        numeric_chars = NEMO_TE_DIGIT | pynini.union(*"0123456789")
        word_char = pynini.difference(
            NEMO_CHAR,
            NEMO_WHITE_SPACE | numeric_chars | pynini.accep('"'),
        )
        word = pynini.closure(word_char, 1)
        spoken_digits = pynini.project(digit, "input").optimize()
        context_word = pynini.difference(word, spoken_digits).optimize()

        def context(category):
            keywords = pynini.project(
                pynini.accep(category) @ context_cues,
                "output",
            ).optimize()

            before = keywords + pynini.closure(retain_separator + context_word, 0, 5)
            after = pynini.closure(context_word + retain_separator, 0, 5) + keywords

            return (
                pynini.closure(before + retain_separator, 0, 1),
                pynini.closure(retain_separator + after, 0, 1),
            )

        def number_field(graph):
            return pynutil.insert('number_part: "') + graph + pynutil.insert('"')

        def sequence(length):
            return digit + pynini.closure(
                delete_separator + digit,
                length - 1,
                length - 1,
            )

        before, after = context("mobile")
        mobile_digits = nonzero + pynini.closure(delete_separator + digit, 9, 9)
        mobile = number_field(before + mobile_digits + after)

        country_code = country_symbol + delete_separator + sequence(2)
        country_field = pynutil.insert('country_code: "') + before + country_code + pynutil.insert('" ')
        mobile |= country_field + delete_separator + number_field(mobile_digits + after)

        before, after = context("landline")
        landline_digits = zero + pynini.closure(delete_separator + digit, 10, 10)
        landline = number_field(before + landline_digits + after)
        before, after = context("pincode")
        pincode = number_field(before + sequence(6) + after)
        before, after = context("credit")
        credit = number_field(before + sequence(4) + after)

        self.final = (
            mobile
            | pynutil.add_weight(landline, 0.01)
            | pynutil.add_weight(credit, 0.02)
            | pynutil.add_weight(pincode, 0.03)
        ).optimize()

        self.fst = self.add_tokens(self.final).optimize()
