#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Jordi Mas i Hernandez <jmas@softcatala.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.

import yaml
import os

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def split_in_two_files(src_filename):

    pairs = set()
    number_validation = 3000
    number_test = 3007 # number_test != number_validation

    strings = 0
    duplicated = 0

    print("Split src and tgt files in 6 files for training, text and validation")

    total_lines = file_len(src_filename)
    validation_each = round(total_lines / number_validation)
    test_each = round(total_lines / number_test)

    if test_each == validation_each:
        print("test_each ({0}) and validation_each  ({0}) cannot be equal".format(test_each, validation_each))
        return
        
    with open("src-val.txt", "w") as source_val,\
        open("src-test.txt", "w") as source_test,\
        open("src-train.txt", "w") as source_train,\
        open(src_filename, "r") as read_source:


        print("total_lines {0}".format(total_lines))
        print("number_validation {0}".format(number_validation))
        print("number_test {0}".format(number_test))
        print("validation_each {0}".format(validation_each))
        print("test_each {0}".format(test_each))

        clean = 0
        while True:

            src = read_source.readline()

            if not src:
                break

            pair = src
            if pair in pairs:
                duplicated = duplicated + 1
                continue
            else:
                pairs.add(pair)

            if strings % validation_each == 0:
                source = source_val
            elif strings % test_each == 0:
                source = source_test
            else:
                source = source_train

            source.write(src)
            strings = strings + 1

    pclean = clean * 100 / strings
    pduplicated = duplicated * 100 / strings
    print(f"Strings: {strings}, duplicated {duplicated} ({pduplicated:.2f}%)")


def append_lines_from_file(src_filename, trg_file):
    lines = 0
    with open(src_filename, 'r') as tf:
        line = tf.readline()
        while line:
            lines += 1
            trg_file.write(line)
            line = tf.readline()

    print("Appended {0} lines from {1}".format(lines, src_filename))
    return lines



def main():

    print("Joins several corpus and creates a final train, validation and test dataset")

    split_in_two_files("corpus/ca_dedup.txt")

if __name__ == "__main__":
    main()
