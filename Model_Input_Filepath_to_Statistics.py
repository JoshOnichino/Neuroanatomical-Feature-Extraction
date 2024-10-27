from Model_Input_to_Masks import JGHPredictor
from Masks_to_Aggregates import overlay_scans, underlay_scans
from Aggregates_to_Filtered_Aggregates import filter_mask
from Filtered_Aggregates_to_Statistics import extract_baseline_features_InHouse

import os
import pandas as pd

from typing import List, Union

def Model_Input_Filepath_to_Masks(input_dir: Union[str, List[List[str]]] = None, 
                                  output_dir: Union[str, List[str]] = None,
                                  model: str = None) -> None:
    # Guarantee the output directory exists
    if not isinstance(input_dir, list):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    use_mirroring = False
    device = 'GPU'
    
    if model == 'TEMP':
        model_folder = r'Dataset005_TemporalisBatch3\nnUNetTrainer__nnUNetPlans__2d'
    elif model == 'TOTAL':
        model_folder = r'Dataset294_TotalSegmentator_part4_muscles_1559subj\nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
    elif model == 'BS':
        model_folder = r'Dataset409_neuro_550subj\nnUNetTrainer_DASegOrd0__nnUNetPlans__3d_fullres_high'
    
    use_folds = (0,)
    checkpoint_name = 'checkpoint_final.pth'    

    predictor = JGHPredictor(use_mirroring, device, model_folder, use_folds, checkpoint_name)
    predictor.option_0001(input_dir, output_dir)

def Masks_to_Aggregates(temp_input: str = None, bs_input: str = None, total_input: str = None,
                        aggregates_output: str = None) -> None:
    
    overlay_scans(total_input, bs_input, aggregates_output)
    underlay_scans(temp_input, aggregates_output, aggregates_output)

def Aggregates_to_Filtered_Aggregates(input: str = None, filtered_output: str = None, classes_to_suppress: list = None) -> None:
    
    filter_mask(input, filtered_output, classes_to_suppress)

def Filtered_Aggregates_to_Statistics(mask_path: str, image_path: str, output_csv: str, cohort: str = 'NSP', site: str = 'JGH', modality: str = 'CT',
                         model_name = 'TotalSegmentatorV2[total, brain_structures]_InHouseTemporalis') -> None:

    patient_id = mask_path.split('_')[0]

    stats = extract_baseline_features_InHouse(image_path, mask_path)

    rows = []

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
            'series_description': os.path.basename(image_path).replace('.nii.gz', ''),
            'volume': features.get('volume', None),
            'mean_density': features.get('mean_density', None),
            'median_density': features.get('median_density', None),
            'std_dev': features.get('std_dev', None),
        }
        rows.append(row)

    # Convert the rows to a DataFrame and save as CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    patient_database = r"C:\Users\joshua.onichino\Dropbox\Head\Patient Database Test"
    patient_id = r"HK20240048470101"
    patient_nifti = r"HK20240048470101.nii.gz"
    patient_nrrd = r"HK20240048470101.nrrd"

    models = ['TEMP', 'BS', 'TOTAL']
    
    for model in models:
        if model == 'TEMP':
            
            print(model)
            
            nrrd_path = os.path.join(patient_database, patient_id, patient_nrrd)
            temp_mask = os.path.join(patient_database, patient_id, f'{patient_id}_{model}')

            print(nrrd_path)
            print(temp_mask)

            Model_Input_Filepath_to_Masks(input_dir=[[nrrd_path]], output_dir=[temp_mask], model=model)
        else:
            nifti_path = os.path.join(patient_database, patient_id, patient_nifti)
            cranial_mask = os.path.join(patient_database, patient_id, f'{patient_id}_{model}')
            
            Model_Input_Filepath_to_Masks(input_dir=[[nifti_path]], output_dir=[cranial_mask], model=model)

    patient_dir = os.path.join(patient_database, patient_id)
    Masks_to_Aggregates(temp_input=os.path.join(patient_dir, f'{patient_id}_TEMP.nrrd'), 
                        bs_input=os.path.join(patient_dir, f'{patient_id}_BS.nii.gz'),
                        total_input=os.path.join(patient_dir, f'{patient_id}_TOTAL.nii.gz'),
                        aggregates_output=os.path.join(patient_dir, f'{patient_id}_AGG.nii.gz'))

    patient_dir = os.path.join(patient_database, patient_id)
    Aggregates_to_Filtered_Aggregates(input=os.path.join(patient_dir, f'{patient_id}_AGG.nii.gz'),
                                      filtered_output=os.path.join(patient_dir, f'{patient_id}_FIL_AGG.nii.gz'),
                                      classes_to_suppress=[22,28])

    patient_dir = os.path.join(patient_database, patient_id)
    Filtered_Aggregates_to_Statistics(mask_path=os.path.join(patient_dir, f'{patient_id}_FIL_AGG.nii.gz'),
                                      image_path=os.path.join(patient_dir, f'{patient_id}.nii.gz'), 
                                      output_csv=os.path.join(patient_dir, 'Aggregated_Statistics.csv'))