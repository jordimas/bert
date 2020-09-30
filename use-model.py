#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="models/roberta/",
    tokenizer="models/roberta/"
)

# The sun <mask>.
# =>

result = fill_mask("La meva mare és <mask>")
