import DCM_to_Model_Input
import Model_Input_to_Masks
import Masks_to_Aggregates
import Aggregates_to_Filtered_Aggregates
import Filtered_Aggregates_to_Statistics

if __name__ == '__main__':
    
    ##### 1 ^ 2 --> Can be Concurrent #####
    
    ##### 3 ^ 4 ^ 5 --> Can be Concurrent #####

    ##### 6 ^ 7 ^ 8 --> Must be Sequential #####

    # 1
    
    input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - DCM"
    output_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - nnUNet - NIfTI"
    desired_format = 'nifti'

    DCM_to_Model_Input.DCM_folder_to_nnUNet(input_dir, output_dir, desired_format)

    # 2

    input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - DCM"
    output_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - nnUNet - NRRD"
    desired_format = 'nrrd'
    
    DCM_to_Model_Input.DCM_folder_to_nnUNet(input_dir, output_dir, desired_format)

    # 3

    input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - nnUNet - NRRD"
    output_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - TEMP"

    Model_Input_to_Masks.NRRD_to_Temporal_Masks(input_dir, output_dir)

    # 4

    input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - nnUNet - NIfTI"
    output_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - TOTAL"

    Model_Input_to_Masks.NIfTI_to_Total_Masks(input_dir, output_dir)

    # 5

    input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - NIfTI"
    output_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - NEURO"

    Model_Input_to_Masks.NIfTI_to_Neuroanatomy_Masks(input_dir, output_dir)

    # 6

    temp_input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - nnUNet - TEMP"
    bs_input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - NEURO"
    total_input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - TOTAL"
    aggregates_output_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - Aggregates"

    Masks_to_Aggregates.Masks_to_Aggregates(temp_input_dir, bs_input_dir, total_input_dir, aggregates_output_dir)

    # 7

    input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - Aggregates"
    filtered_output_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - Aggregates - FILTERED"
    classes_to_suppress = [22, 28]

    Aggregates_to_Filtered_Aggregates.Aggregates_to_Filtered_Aggregates(input_dir, filtered_output_dir, classes_to_suppress)

    # 8

    mask_folder = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - Aggregates - FILTERED"
    image_folder = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - nnUNet - NIfTI"
    output_csv = r"C:\Users\joshua.onichino\Dropbox\Head\Test\Batch 5 - STATISTICS"

    Filtered_Aggregates_to_Statistics.Filtered_Aggregates_to_Statistics(mask_folder, image_folder, output_csv)