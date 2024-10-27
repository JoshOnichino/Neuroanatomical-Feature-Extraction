import os
import numpy as np
import SimpleITK as sitk

def Aggregates_to_Filtered_Aggregates(input_dir: str = None, filtered_output_dir: str = None, classes_to_suppress: list = None) -> None:
    # Guarantee that the output path exists
    if not os.path.exists(filtered_output_dir):
        os.mkdir(filtered_output_dir)

    # Iterate over all files in the folder
    for filename in os.listdir(input_dir):
        if os.path.exists(os.path.join(filtered_output_dir, filename)):
            continue

        filter_mask(os.path.join(input_dir, filename), os.path.join(filtered_output_dir, filename), classes_to_suppress)

def filter_mask(mask_path: str, output_path: str, classes_to_suppress: list = None):
    """
    Filters a segmentation mask by:
    1. Suppressing classes (setting them to background) if specified.
    2. Ensuring class labels are continuous (no gaps between classes).
    
    Parameters:
    - mask_path: str, path to the segmentation mask file (e.g., .nii.gz or .nrrd).
    - output_path: str, path to save the modified mask.
    - classes_to_suppress: list of int (optional), the classes that need to be suppressed (set to background).
    
    Returns:
    - filtered_mask: np.ndarray, the modified mask after suppression and class rearrangement.
    """
    # Read the mask image using SimpleITK
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    
    # Step 1: Suppress specified classes if provided
    if classes_to_suppress:
        for cls in classes_to_suppress:
            mask_arr[mask_arr == cls] = 0
    
    # Step 2: Ensure the classes are continuous (no gaps)
    # Get the unique class labels in the mask, sorted
    unique_classes = sorted(np.unique(mask_arr))
    
    # Initialize a dictionary to map the old classes to new continuous classes
    class_mapping = {}
    current_label = 1  # Start with 1 as the first label

    for cls in unique_classes:
        if cls == 0:
            continue  # Skip the background class
        
        class_mapping[cls] = current_label
        current_label += 1

    # Apply the mapping to the mask
    filtered_mask = np.copy(mask_arr)
    for old_class, new_class in class_mapping.items():
        filtered_mask[mask_arr == old_class] = new_class
    
    # Convert the modified array back to a SimpleITK image
    filtered_mask_img = sitk.GetImageFromArray(filtered_mask)
    filtered_mask_img.CopyInformation(mask_img)  # Preserve spatial information from the original image

    # Save the filtered mask to the specified output path
    sitk.WriteImage(filtered_mask_img, output_path, useCompression=True)
    
    return filtered_mask

if __name__ == '__main__':
    input_dir = r"C:\Users\josho\Dropbox\Head\Batch 5 - Aggregates - FINAL"
    filtered_output_dir = r"C:\Users\josho\Dropbox\Head\Batch 5 - Aggregates - FILTERED"
    classes_to_suppress = [22, 28]
    Aggregates_to_Filtered_Aggregates(input_dir, filtered_output_dir, classes_to_suppress)