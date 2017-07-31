#!/bin/bash

INPUTXYZFOLDER=$WRKDIR/varmad1/MechAFM/myCode/FINAL/inputXYZ
OUTPUTXYZ=$WRKDIR/varmad1/MechAFM/myCode/FINAL/outputxyz
DOSTUFF=$WRKDIR/varmad1/MechAFM/myCode/dostuff.py
cd $INPUTXYZFOLDER
for i in *.xyz; do
    INPUTXYZ=$INPUTXYZFOLDER/$i
    python $DOSTUFF -i $INPUTXYZ -o $OUTPUTXYZ
    cp $INPUTXYZFOLDER/template.job $OUTPUTXYZ/${i%.xyz}
    cd $OUTPUTXYZ/${i%.xyz}
    sbatch ./template.job
    cd ..
done
