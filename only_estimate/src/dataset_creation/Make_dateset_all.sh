names="matumoto yoshikura takahashi takahashi_jr taniguchi kawai goto togo gosha asano mori sato"
target_num="1 2 3 4 5 6 7 8 9 10 11 12"

for i in $target_num; do
    echo $i | python3 Make_dataset_0120.py
done