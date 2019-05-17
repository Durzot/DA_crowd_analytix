#!bin/bash

all_max_depth=$(seq 10 5 40)
all_colsample_bytree=$(seq 0.3 0.2 1)
all_subsample=$(seq 0.4 0.2 1)
all_eta=$(seq 0.05 0.4 1)

for max_depth in ${all_max_depth}
do
    for colsample_bytree in ${all_colsample_bytree}
    do
        for subsample in ${all_subsample}
        do
            pids=""
            for eta in ${all_eta}
            do
                echo "Processing max_depth ${max_depth} colsample bytree
                ${colsample_bytree}"
                python ../source_2/run_models/run_xgb.py --model_type XGB_3 --max_depth ${max_depth} --colsample_bytree ${colsample_bytree} --subsample ${subsample} --eta ${eta} &> /dev/null $
                pids+=" $!"
            done

            for pid in ${pids}
            do 
                if wait ${pid}
                then 
                    echo "${pid} eta finished"
                else
                    echo "${pid} eta finished"
                    exit 1
                fi
            done
        done
    done
done

