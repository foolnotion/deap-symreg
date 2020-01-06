for i in ./data/*.json
do
    CSVFILE=`echo "$i" | awk -F'/' '{ print $3 }' | sed 's/json/csv/g'`
    seq 1 50 | parallel --bar -N0 python symreg.py --data "${i}" > runs/"${CSVFILE}"
done

