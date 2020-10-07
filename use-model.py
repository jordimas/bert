#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="models/roberta/output",
    tokenizer="models/roberta/"
)

# The sun <mask>.
# =>

result = fill_mask("El meu cotxe és molt millor del que molts <mask>.")
print(result)
