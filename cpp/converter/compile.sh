#!/bin/bash

rootcling -f VectorDict.cxx ROOT2CSVconverter.hpp LinkDef.hpp

g++ VectorDict.cxx ROOT2CSVconverter.cpp -o ROOT2CSVconverter `root-config --cflags --glibs`
