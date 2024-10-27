import os
import SimpleITK as sitk
from typing import Union
import shutil

def DCM_folder_to_nnUNet(input_dir: str = None, nnUNet_output_dir: str = None, n_inputs: int = 0, desired_format: str = None) -> None:
    # Guarantee that the output path exists
    if not os.path.exists(nnUNet_output_dir):
        os.mkdir(nnUNet_output_dir)
    
    if desired_format == 'nrrd':
        file_ending = 'nrrd'
        intermediate_output_dir = f'{input_dir} - NRRD'
        if not os.path.exists(intermediate_output_dir):
            os.mkdir(intermediate_output_dir)
    elif desired_format == "nifti":
        file_ending == 'nii.gz'
        intermediate_output_dir = f'{input_dir} - NIfTI'
        if not os.path.exists(intermediate_output_dir):
            os.mkdir(intermediate_output_dir)

    # Iterate over all files in the folder
    for filename in os.listdir(input_dir):
        if os.path.exists(os.path.join(intermediate_output_dir, filename)): # MAY NEED TO BE ADAPTED FOR MULTIPLE INPUTS
            if os.path.exists(os.path.join(intermediate_output_dir, filename,  f'{filename}.{file_ending}')):
                continue
        
        dcm_dir = os.path.join(input_dir, filename, 'DICOM', 'EXP00000')
        convert_DCM_to_desired_format(dcm_dir, intermediate_output_dir, desired_format, filename)

    for filename in os.listdir(intermediate_output_dir):
        if os.path.exists(os.path.join(nnUNet_output_dir, f'{filename.split(".")[0]}_0000.{file_ending}')): # MAY NEED TO BE ADAPTED FOR MULTIPLE INPUTS
            continue

        original_file = os.path.join(intermediate_output_dir, filename, f'{filename}.{file_ending}')
        to_nnUNet_name(original_file, nnUNet_output_dir, file_ending) # MAY NEED TO BE ADAPTED FOR MULTIPLE INPUTS

def convert_DCM_to_desired_format(dcm_dir: str = None, output_dir: str = None, desired_format: str = None, case_name: str = None) -> Union[None, int]:
    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(dcm_series)
    
    # Load the DICOM series into an image
    image = reader.Execute()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Make a subdirectory for the case
    subdir = os.path.join(output_dir, case_name)
    if not os.path.exists(subdir):
        os.mkdir(subdir)  

    # Save the image
    if desired_format == 'nrrd':
        if not os.path.exists(os.path.join(subdir, f'{case_name}.nrrd')):
            sitk.WriteImage(image, os.path.join(subdir, f'{case_name}.nrrd'))
        else:
            return -1
    elif desired_format == 'nifti':
        if not os.path.exists(os.path.join(subdir, f'{case_name}.nii.gz')):
            sitk.WriteImage(image, os.path.join(subdir, f'{case_name}.nii.gz'))
        else:
            return -1

def to_nnUNet_name(input_file: str = None, nnUNet_output_dir: str = None, file_ending: str = None) -> None:
    filename = os.path.basename(input_file)

    if not os.path.exists(nnUNet_output_dir):
        os.mkdir(nnUNet_output_dir)
    nnUNet_file = os.path.join(nnUNet_output_dir, f'{filename.split(".")[0]}_0000.{file_ending}')

    if not os.path.exists(nnUNet_file):
        shutil.copy2(input_file, nnUNet_file)
    else:
        return -1


if __name__ == '__main__':

    input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - DCM"
    output_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - DCM - NIfTI"