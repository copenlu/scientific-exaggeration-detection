#!/bin/bash

wget https://raw.githubusercontent.com/junwang4/correlation-to-causation-exaggeration/master/data/annotated_eureka.csv
wget https://raw.githubusercontent.com/junwang4/correlation-to-causation-exaggeration/master/data/annotated_pubmed.csv

python combine_strength_annotation.py