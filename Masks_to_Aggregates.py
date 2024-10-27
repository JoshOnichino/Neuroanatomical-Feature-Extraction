import SimpleITK as sitk
import numpy as np
import os

def Masks_to_Aggregates(temp_input_dir: str = None, bs_input_dir: str = None, total_input_dir: str = None,
                        aggregates_output_dir: str = None) -> None:
    
    # Ensure the output directory exists
    if not os.path.exists(aggregates_output_dir):
        os.mkdir(aggregates_output_dir)

    # Iterate over all files in the total input directory
    for filename in os.listdir(total_input_dir):
        # Check if the file is either an .nrrd or .nii.gz file
        if not (filename.endswith('.nrrd') or filename.endswith('.nii.gz')):
            continue
        
        # Check if the output file already exists to avoid redundant work
        if os.path.exists(os.path.join(aggregates_output_dir, filename)):
            continue

        total_file = os.path.join(total_input_dir, filename)
        bs_file = os.path.join(bs_input_dir, filename)
        
        # Call the overlay_scans function with the two files and the output directory
        overlay_scans(total_file, bs_file, os.path.join(aggregates_output_dir, filename))
        
    # Ensure the output directory exists
    final_aggregates_output_dir = f'{aggregates_output_dir} - FINAL'
    if not os.path.exists(final_aggregates_output_dir):
        os.mkdir(final_aggregates_output_dir)

    # Iterate over all files in the aggregates output directory
    for filename in os.listdir(aggregates_output_dir):

        # Check if the output file already exists to avoid redundant work
        if os.path.exists(os.path.join(final_aggregates_output_dir, filename)):
            continue

        temp_file = os.path.join(temp_input_dir, f'{filename.split(".")[0]}.nrrd')
        aggregate_file = os.path.join(aggregates_output_dir, filename)
        
        # Call the overlay_scans function with the two files and the output directory
        underlay_scans(temp_file, aggregate_file, os.path.join(final_aggregates_output_dir, filename))

def overlay_scans(send_to_back_path: str, overlay_scan_path: str, output_path: str):
    # Load the segmentation scans as SimpleITK images
    send_to_back = sitk.ReadImage(send_to_back_path)
    overlay_scan = sitk.ReadImage(overlay_scan_path)
    
    # Ensure both images have the same orientation, spacing, and origin
    overlay_scan = sitk.Resample(overlay_scan, send_to_back, sitk.Transform(), sitk.sitkNearestNeighbor, 0, overlay_scan.GetPixelID())
    
    # Convert the scans to numpy arrays for easier manipulation
    send_to_back_array = sitk.GetArrayFromImage(send_to_back)
    overlay_scan_array = sitk.GetArrayFromImage(overlay_scan)
    
    # Identify unique classes in both scans
    back_classes = [cls for cls in np.unique(send_to_back_array) if cls != 0]
    overlay_classes = [cls for cls in np.unique(overlay_scan_array) if cls != 0]
    
    # Find overlapping classes
    overlapping_classes = np.intersect1d(back_classes, overlay_classes)
    
    # If overlapping classes exist, shift the `send_to_front` class numbers to avoid conflicts
    if overlapping_classes.size > 0:
        max_overlay_class = np.max(overlay_classes)
        shift_amount = max_overlay_class + 1  # Shift `send_to_front` classes to avoid overlap
        
        for cls in overlapping_classes:
            # Shift the class numbers in send_to_front to new, non-overlapping values
            send_to_back_array[send_to_back_array == cls] += shift_amount

    # Combine the scans: wherever the overlay_scan has a non-zero label, it takes precedence
    combined_array = np.where(overlay_scan_array != 0, overlay_scan_array, send_to_back_array)
    
    # Convert the result back to a SimpleITK image
    combined_image = sitk.GetImageFromArray(combined_array)
    
    # Copy the metadata (spacing, origin, direction) from the send_to_back scan
    combined_image.CopyInformation(send_to_back)
    
    # Write the combined result to an output file
    sitk.WriteImage(combined_image, output_path, useCompression=True)

def underlay_scans(send_to_front_path: str, underlay_scan_path: str, output_path: str) -> None:
    # Load the segmentation scans as SimpleITK images
    send_to_front = sitk.ReadImage(send_to_front_path)
    underlay_scan = sitk.ReadImage(underlay_scan_path)
    
    # Ensure both images have the same orientation, spacing, and origin
    underlay_scan = sitk.Resample(underlay_scan, send_to_front, sitk.Transform(), sitk.sitkNearestNeighbor, 0, underlay_scan.GetPixelID())
    
    # Convert the scans to numpy arrays for easier manipulation
    send_to_front_array = sitk.GetArrayFromImage(send_to_front)
    underlay_scan_array = sitk.GetArrayFromImage(underlay_scan)
        
    # Identify unique classes in both scans
    front_classes = [cls for cls in np.unique(send_to_front_array) if cls != 0]
    underlay_classes = [cls for cls in np.unique(underlay_scan_array) if cls != 0]
    
    # Find overlapping classes
    overlapping_classes = np.intersect1d(front_classes, underlay_classes)
    
    # If overlapping classes exist, shift the `send_to_front` class numbers to avoid conflicts
    if overlapping_classes.size > 0:
        max_underlay_class = np.max(underlay_classes)
        shift_amount = max_underlay_class + 1  # Shift `send_to_front` classes to avoid overlap
        
        for cls in overlapping_classes:
            # Shift the class numbers in send_to_front to new, non-overlapping values
            send_to_front_array[send_to_front_array == cls] += shift_amount

    # Combine the scans: wherever the overlay_scan has a non-zero label, it takes precedence
    combined_array = np.where(send_to_front_array != 0, send_to_front_array, underlay_scan_array)
    
    # Convert the result back to a SimpleITK image
    combined_image = sitk.GetImageFromArray(combined_array)
    
    # Copy the metadata (spacing, origin, direction) from the `send_to_front` scan
    combined_image.CopyInformation(send_to_front)
    
    # Write the combined result to an output file
    sitk.WriteImage(combined_image, output_path, useCompression=True)

if __name__ == '__main__':
    temp_input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - nnUNet - TEMP"
    bs_input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - Cranial Anatomy 1"
    total_input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - Cranial Anatomy 0"
    aggregates_output_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - Aggregates"

    Masks_to_Aggregates(temp_input_dir, bs_input_dir, total_input_dir, aggregates_output_dir)