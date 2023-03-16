for ct in THCA SARC
do
    echo "Running $ct"
    if (( $ct == "THCA" ))
    then
        ns=4
        config="TCGA/model_config_files/Sagittarius_deeper_config.json"
    elif (( $ct == "SARC" )) 
    then
        ns=2
        config="TCGA/model_config_files/Sagittarius_config.json"
    fi
    python TCGA/Sagittarius_patient_by_patient_extrapolation.py --cancer-type $ct --non-stationary $ns --config $config --seed 0 --model-file "TCGA_test/TCGA_"$ct"_model.pth"
done