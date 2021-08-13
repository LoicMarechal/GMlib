#!/bin/sh

if [ $# -ne 2 ]
then
	echo "gmlpart source.meshb nb_parts"
	exit 1
fi

inputbase=`basename $1 .meshb`
metismesh=${inputbase}_metis.mesh
outname=${inputbase}$2"parts"

if [ ! -f "$metismesh" ]
then
   echo "convert" $1 "to" $metismesh
   meshb2metis -in $1 -out ${inputbase}
fi

mpmetis -gtype=nodal ${metismesh} $2
hilbert -in $1 -out $outname -gmlib -ndom $2 -npart ${metismesh}.npart.$2 -epart ${metismesh}.epart.$2 8
