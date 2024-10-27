import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
import pandas as pd
import os

def Filtered_Aggregates_to_Statistics(mask_folder: str, image_folder: str, output_csv: str, cohort: str = 'NSP', site: str = 'JGH', modality: str = 'CT',
                         model_name = 'TotalSegmentatorV2[total, brain_structures]_InHouseTemporalis') -> None:
    """
    Iterates through the image and mask folders, extracts statistics for each segment,
    aggregates them into a dictionary, and saves them into a single CSV file.
    
    Parameters:
    - mask_folder: str, path to the folder containing masks.
    - image_folder: str, path to the folder containing corresponding images.
    - output_csv: str, path to save the aggregated statistics as a CSV file.
    - cohort: str, the cohort name (default 'NSP').
    - site: str, the site name (default 'JGH').
    - modality: str, the modality type (default 'CT').
    """
    # Define the list to hold all records for the CSV
    rows = []
    
    # Iterate through each mask file in the mask folder
    for mask_file in os.listdir(mask_folder):
        if mask_file.endswith('.nii.gz'):
            # Extract the base patient ID from the mask file (removing the extension)
            patient_id = mask_file.split('.')[0]
            mask_path = os.path.join(mask_folder, mask_file)
            
            # Locate the corresponding image file based on the patient ID
            image_file = f"{patient_id}_0000.nii.gz"
            image_path = os.path.join(image_folder, image_file)
            
            # Check if the image file exists
            if not os.path.exists(image_path):
                print(f"Image for patient {patient_id} not found! Skipping...")
                continue
            
            # Extract statistics for each segment in the mask
            stats = extract_baseline_features_InHouse(image_path, mask_path)
            
            # Create rows for each segment in the mask
            for segment_id, features in stats.items():
                row = {
                    'segment': segment_id,
                    'segment_feature': 'size',  # Placeholder, can be updated as needed
                    'cohort': cohort,
                    'site': site,
                    'modality': modality,
                    'model_name': model_name,
                    'patient_id': patient_id,
                    'series_description': os.path.basename(image_path).replace('_0000.nii.gz', ''),
                    'volume': features.get('volume', None),
                    'mean_density': features.get('mean_density', None),
                    'median_density': features.get('median_density', None),
                    'std_dev': features.get('std_dev', None),
                }
                rows.append(row)

    # Convert the rows to a DataFrame and save as CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

def extract_baseline_features_PyRads(image_path, mask_path):
    """
    Extracts baseline statistics (volume, surface area, mean intensity, std intensity, and median intensity)
    for each segment in the mask using PyRadiomics.
    
    Parameters:
    - image_path: str, path to the input image (e.g., .nii.gz).
    - mask_path: str, path to the segmentation mask file (e.g., .nii.gz).
    
    Returns:
    - result_dict: dict, containing statistics for each segment.
    """
    # Initialize the PyRadiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    
    # Disable all features except for the basics we're interested in
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')

    # Specify the features we want from PyRadiomics
    extractor.enableFeaturesByName(firstorder=['Mean', 'Median', 'RobustMeanAbsoluteDeviation'])
    extractor.enableFeaturesByName(shape=['VoxelVolume'])
    
    # Create an empty dictionary to store results for each segment
    result_dict = {}
    
    # Read the segmentation mask
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    unique_segments = [int(x) for x in set(mask_arr.flatten()) if x != 0]  # Get unique non-zero segment IDs
    
    # Loop through each segment and extract features
    for segment_id in unique_segments:
        segment_specific_mask = sitk.BinaryThreshold(mask_img, segment_id, segment_id, 1, 0)
        # Execute feature extraction for the specific segment
        result = extractor.execute(image_path, segment_specific_mask)
        
        # Gather the relevant statistics for this segment
        features = {
            'volume': result.get(f'original_shape_VoxelVolume', None),
            'mean_density': result.get(f'original_firstorder_Mean', None),
            'median_density': result.get(f'original_firstorder_Median', None),
            'robust_mad': result.get(f'original_firstorder_RobustMeanAbsoluteDeviation', None)
        }
        
        result_dict[segment_id] = features

    return result_dict

def extract_baseline_features_InHouse(image_path, mask_path):
    """
    Extracts baseline statistics (volume, surface area, mean intensity, median intensity, and standard deviation)
    for each segment in the mask by directly calculating the parameters without relying on external libraries.
    
    Parameters:
    - image_path: str, path to the input image (e.g., .nii.gz).
    - mask_path: str, path to the segmentation mask file (e.g., .nii.gz).
    
    Returns:
    - result_dict: dict, containing statistics for each segment.
    """
    
    # Read the image and mask
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    
    # Get the numpy arrays
    image_arr = sitk.GetArrayFromImage(image)  # Shape: [z, y, x]
    mask_arr = sitk.GetArrayFromImage(mask)
    
    # Get image spacing to calculate physical dimensions
    spacing = image.GetSpacing()  # (dx, dy, dz)
    dx, dy, dz = spacing
    voxel_volume = dx * dy * dz  # Volume of one voxel
    
    # Get unique segment IDs excluding background (assuming background is 0)
    unique_segments = np.unique(mask_arr)
    unique_segments = unique_segments[unique_segments != 0]
    
    result_dict = {}
    
    for segment_id in unique_segments:
        # Create a binary mask for the segment
        segment_mask = (mask_arr == segment_id)
        
        # Calculate volume
        num_voxels = np.sum(segment_mask)
        volume = num_voxels * voxel_volume  # in physical units (e.g., mm^3)
        
        # Extract intensities within the segment
        segment_intensities = image_arr[segment_mask]
        
        # Compute mean, median, and standard deviation
        mean_density = np.mean(segment_intensities)
        median_density = np.median(segment_intensities)
        std_dev = np.std(segment_intensities)
        
        # Store the calculated features in the result dictionary
        features = {
            'volume': volume,
            'mean_density': mean_density,
            'median_density': median_density,
            'std_dev': std_dev
        }
        
        result_dict[segment_id] = features
    
    return result_dict

if __name__ == "__main__":
    mask_folder = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - Aggregates - FILTERED - Copy"
    image_folder = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - nnUNet - TOTAL"
    output_csv =r"C:\Users\joshua.onichino\Dropbox\Head\Aggregated_Statistics_TEST.csv"
    Filtered_Aggregates_to_Statistics(mask_folder, image_folder, output_csv)