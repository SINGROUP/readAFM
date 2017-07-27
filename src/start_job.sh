for SIGMAZ in 15.0 25.0 50.0 100.0; do
    cat parameters.in | sed s/SIGMAZ/$SIGMAZ/  > parameters$SIGMAZ.in
    sbatch run_minimalCNN_gpu.slrm $SIGMAZ
done
