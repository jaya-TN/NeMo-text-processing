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
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing Telugu dates.
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        month = pynutil.delete('month: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        day = pynutil.delete('day: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        year = pynutil.delete('year: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        text = pynutil.delete('text: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        era = pynutil.delete('era: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        year_prefix = pynutil.delete('year_prefix: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        year_suffix = pynutil.delete('year_suffix: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        graph_day = day

        graph_day_month = day + delete_extra_space + month

        graph_month_day = month + delete_extra_space + day

        graph_day_month_year = day + delete_extra_space + month + pynutil.insert(",") + delete_extra_space + year

        graph_month_day_year = month + delete_extra_space + day + pynutil.insert(",") + delete_extra_space + year

        graph_month_year = month + delete_extra_space + year

        graph_day_text = day + delete_extra_space + text

        graph_month_day_text = month + delete_extra_space + day + delete_extra_space + text

        graph_year_text = year + delete_extra_space + text

        graph_text_year = text + delete_extra_space + year

        graph_text_year_text = text + delete_extra_space + year + delete_extra_space + text

        graph_day_month_year_text = (
            day
            + delete_extra_space
            + month
            + pynutil.insert(",")
            + delete_extra_space
            + year
            + delete_extra_space
            + text
        )

        graph_month_year_text = month + delete_extra_space + year + delete_extra_space + text

        graph_year_text_text = year + delete_extra_space + text + delete_extra_space + text

        graph_year_era = year + delete_extra_space + era

        graph_day_month_year_era = (
            day
            + delete_extra_space
            + month
            + pynutil.insert(",")
            + delete_extra_space
            + year
            + delete_extra_space
            + era
        )

        graph_month_year_era = month + pynutil.insert(",") + delete_extra_space + year + delete_extra_space + era

        graph_prefixed_year = year_prefix + delete_extra_space + year

        graph_prefixed_year_era = year_prefix + delete_extra_space + year + delete_extra_space + era

        graph_year_suffix = year + delete_space + year_suffix

        graph_day_month_year_suffix = (
            day
            + delete_extra_space
            + month
            + pynutil.insert(",")
            + delete_extra_space
            + year
            + delete_space
            + year_suffix
        )

        optional_preserve_order = pynini.closure(
            delete_space + pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true"),
            0,
            1,
        )

        graph = (
            graph_day
            | graph_day_month
            | graph_month_day
            | graph_day_month_year
            | graph_month_day_year
            | graph_month_year
            | graph_day_text
            | graph_month_day_text
            | graph_year_text
            | graph_text_year
            | graph_text_year_text
            | graph_day_month_year_text
            | graph_month_year_text
            | graph_year_text_text
            | graph_year_era
            | graph_day_month_year_era
            | graph_month_year_era
            | graph_prefixed_year
            | graph_prefixed_year_era
            | graph_year_suffix
            | graph_day_month_year_suffix
            | year
        )

        self.fst = self.delete_tokens(graph + optional_preserve_order).optimize()
