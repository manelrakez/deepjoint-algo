ROOT="specify/your/root/path"

directories=("directory_name")
folder_name="Splits_data"

# Iterate through the list of directories
for directory in "${directories[@]}"
do
    # Check if the folder exists
    if [ ! -d "${ROOT}/$directory/$folder_name" ]; then
        # If not, create it
        mkdir -p "${ROOT}/$directory/$folder_name"
        echo "Folder '$folder_name' created in '$directory'."
    else
        echo "Folder '$folder_name' already exists in '$directory'."
    fi
done

Rscript "${ROOT}/directory_name/01_ModelFit_LeftTrunc_DAorPD_CVCS.R"