#!/bin/bash

RESNETNAME="resnet18"

BSVALUE="128"
BSNAME="bs${BSVALUE}"

PS="0.0"
WD="0.0"

DATADIR="/volume01/data"
BASEOUTDIR="/volume01/output"
OUTDIR="$BASEOUTDIR/nuclear_paper_code/revision01/$RESNETNAME/$BSNAME"
