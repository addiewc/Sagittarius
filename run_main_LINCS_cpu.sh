for taskname in "complete_generation" "combination_and_dose" "combination_and_time"
do
    echo "Running $taskname"
    python "LINCS/Sagittarius_LINCS_"$taskname"_experiment.py" --config LINCS/model_config_files/Sagittarius_config.json --seed 0 --model-file "LINCS_test/LINCS_"$taskname"_model.pth"
done