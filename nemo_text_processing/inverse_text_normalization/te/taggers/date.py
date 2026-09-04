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

from nemo_text_processing.inverse_text_normalization.te.graph_utils import NEMO_TE_DIGIT, GraphFst, delete_space
from nemo_text_processing.inverse_text_normalization.te.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classifying Telugu dates.
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst):
        super().__init__(name="date", kind="classify")

        graph_ordinal = ordinal.graph_no_exception

        graph_day = pynini.string_file(get_abs_path("data/date/days.tsv"))
        graph_month = pynini.string_file(get_abs_path("data/date/months.tsv"))
        graph_ordinal_day = pynini.string_file(get_abs_path("data/date/ordinal_days.tsv"))
        graph_era = pynini.string_file(get_abs_path("data/date/era.tsv"))
        graph_century = pynini.string_file(get_abs_path("data/date/century.tsv"))
        graph_range = pynini.string_file(get_abs_path("data/date/range.tsv"))
        graph_date_marker = pynini.string_file(get_abs_path("data/date/date_marker.tsv"))
        graph_year_prefix = pynini.string_file(get_abs_path("data/date/year_prefix.tsv"))
        graph_year_suffix = pynini.string_file(get_abs_path("data/date/year_suffix.tsv"))
        graph_year_suffix_attached = pynini.string_file(get_abs_path("data/date/year_suffix_attached.tsv"))

        # Handles attached Telugu day suffixes generically.

        graph_day_suffix = (graph_day + pynini.union("న", "కి")).optimize()

        graph_year = cardinal.graph_no_exception @ pynini.closure(
            NEMO_TE_DIGIT,
            1,
            4,
        )

        graph_short_year = cardinal.graph_no_exception @ pynini.closure(
            NEMO_TE_DIGIT,
            1,
            2,
        )

        day = pynutil.insert('day: "') + graph_day + pynutil.insert('" ')

        day_suffix = pynutil.insert('day: "') + graph_day_suffix + pynutil.insert('" ')

        ordinal_day = pynutil.insert('day: "') + graph_ordinal_day + pynutil.insert('" ')

        ordinal_suffix = pynini.accep("వ")

        ordinal_century = pynutil.insert('day: "') + graph_ordinal + ordinal_suffix + pynutil.insert('" ')
        month = pynutil.insert('month: "') + graph_month + pynutil.insert('" ')
        year = pynutil.insert('year: "') + graph_year + pynutil.insert('" ')
        era = pynutil.insert('era: "') + graph_era + pynutil.insert('" ')
        century = pynutil.insert('text: "') + graph_century + pynutil.insert('" ')
        date_marker = pynutil.insert('text: "') + graph_date_marker + pynutil.insert('" ')
        year_prefix = pynutil.insert('year_prefix: "') + graph_year_prefix + pynutil.insert('" ')
        year_suffix = pynutil.insert('text: "') + graph_year_suffix + pynutil.insert('" ')

        year_suffix_attached = pynutil.insert('year_suffix: "') + graph_year_suffix_attached + pynutil.insert('" ')

        graph_day_month = day + delete_space + month

        graph_month_day = month + delete_space + day + pynutil.insert(" preserve_order: true")

        graph_day_month_year = day + delete_space + month + delete_space + year

        graph_month_day_year = (
            month + delete_space + day + delete_space + year + pynutil.insert(" preserve_order: true")
        )

        graph_month_year = month + delete_space + year

        graph_day_marker = day + delete_space + date_marker

        graph_ordinal_day_marker = ordinal_day + delete_space + date_marker

        graph_month_ordinal_day = (
            month + delete_space + ordinal_day + delete_space + date_marker + pynutil.insert(" preserve_order: true")
        )

        graph_year_only = year

        graph_year_era = year + delete_space + era

        graph_day_month_year_era = day + delete_space + month + delete_space + year + delete_space + era

        graph_month_year_era = month + delete_space + year + delete_space + era

        graph_year_range = (
            pynutil.insert('year: "')
            + graph_year
            + delete_space
            + graph_range
            + delete_space
            + graph_year
            + pynutil.insert('" ')
        )

        graph_year_short_range = (
            pynutil.insert('year: "')
            + graph_year
            + delete_space
            + graph_range
            + delete_space
            + graph_short_year
            + pynutil.insert('" ')
            + delete_space
            + year_suffix
        )

        graph_year_range_era = graph_year_range + delete_space + era

        graph_prefixed_year = year_prefix + delete_space + year + pynutil.insert(" preserve_order: true")

        graph_prefixed_year_era = (
            year_prefix + delete_space + year + delete_space + era + pynutil.insert(" preserve_order: true")
        )

        graph_suffixed_year = year + delete_space + year_suffix

        graph_attached_suffixed_year = year + delete_space + year_suffix_attached

        graph_day_month_attached_suffixed_year = (
            day + delete_space + month + delete_space + year + delete_space + year_suffix_attached
        )

        graph_ordinal_century = ordinal_century + delete_space + century

        graph_cardinal_century = year + delete_space + century

        graph = (
            graph_day_month
            | graph_month_day
            | graph_day_month_year
            | graph_month_day_year
            | graph_month_year
            | graph_day_marker
            | graph_ordinal_day_marker
            | graph_month_ordinal_day
            | day_suffix
            | graph_year_only
            | graph_year_era
            | graph_day_month_year_era
            | graph_month_year_era
            | graph_year_range
            | graph_year_short_range
            | graph_year_range_era
            | graph_prefixed_year
            | graph_prefixed_year_era
            | graph_suffixed_year
            | graph_attached_suffixed_year
            | graph_day_month_attached_suffixed_year
            | graph_ordinal_century
            | graph_cardinal_century
        )

        self.fst = self.add_tokens(graph).optimize()
