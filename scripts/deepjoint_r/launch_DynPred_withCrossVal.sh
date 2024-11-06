ROOT="specify/your/root/path"

# Check if folder exists
directory = ("folder")
if [ ! -d "${ROOT}/$directory" ]; then
    mkdir -p "${ROOT}/$directory"
    echo "Folder 'folder' created in '${ROOT}'."
else
    echo "Folder 'folder' already exists in '${ROOT}'."
fi

directories=("Pred_fold_1" "Pred_fold_2" "Pred_fold_3" "Pred_fold_4" "Pred_fold_5")
# Iterate through the list of directories
for directory in "${directories[@]}"
do
    # Check if the folder exists
    if [ ! -d "${ROOT}/folder/$directory" ]; then
        # If not, create it
        mkdir -p "${ROOT}/folder/$directory"
        echo "Folder '$directory' created in folder."
    else
        echo "Folder '$directory' already exists in folder."
    fi
done

#1st step: Fit the model to 5-fold train datasets
Rscript "${ROOT}/03_1_DynPred_withCrossVal_ModelFit_Fold1.R"
Rscript "${ROOT}/03_2_DynPred_withCrossVal_ModelFit_Fold2.R"
Rscript "${ROOT}/03_3_DynPred_withCrossVal_ModelFit_Fold3.R"
Rscript "${ROOT}/03_4_DynPred_withCrossVal_ModelFit_Fold4.R"
Rscript "${ROOT}/03_5_DynPred_withCrossVal_ModelFit_Fold5.R"

#2nd step: Calculate breast cancer risk predictions for different landmark times
Rscript "${ROOT}/04_DynPred_withCrossVal_PredictionsCalculation.R"

#3rd step: Compute AUC and BS
Rscript "${ROOT}/05_DynPred_withCrossVal_AUCandBSCalculation.R"