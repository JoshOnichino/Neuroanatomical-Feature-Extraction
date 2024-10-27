from nnunetv2.paths import nnUNet_results
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import os
from typing import Tuple, Union, List

def NRRD_to_Temporal_Masks(input_dir: str = None, temp_output_dir: str = None) -> None:

    if not os.path.exists(temp_output_dir):
        os.mkdir(temp_output_dir)

    use_mirroring = True
    device = 'GPU'
    
    model_folder = r'Dataset005_TemporalisBatch3\nnUNetTrainer__nnUNetPlans__2d'
    use_folds = (0,1,2,3,4)
    checkpoint_name = 'checkpoint_final.pth'

    predictor = JGHPredictor(use_mirroring, device, model_folder, use_folds, checkpoint_name)
    predictor.option_0001(input_dir, temp_output_dir)

def NIfTI_to_Total_Masks(input_dir: str = None, total_output_dir: str = None) -> None:
    if not os.path.exists(total_output_dir):
        os.mkdir(total_output_dir)

    use_mirroring = False
    device = 'GPU'
    model_folder = r"Dataset294_TotalSegmentator_part4_muscles_1559subj\nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
    use_folds = (0,)
    checkpoint_name = 'checkpoint_final.pth'

    predictor = JGHPredictor(use_mirroring, device, model_folder, use_folds, checkpoint_name)
    predictor.option_0001(input_dir, output_dir)

def NIfTI_to_Neuroanatomy_Masks(input_dir: str = None, total_output_dir: str = None) -> None:
    if not os.path.exists(total_output_dir):
        os.mkdir(total_output_dir)

    use_mirroring = False
    device = 'GPU'
    model_folder = r"Dataset409_neuro_550subj\nnUNetTrainer_DASegOrd0__nnUNetPlans__3d_fullres_high"
    use_folds = (0,)
    checkpoint_name = 'checkpoint_final.pth'

    predictor = JGHPredictor(use_mirroring, device, model_folder, use_folds, checkpoint_name)
    predictor.option_0001(input_dir, output_dir)

class JGHPredictor(object):
    def __init__(self, 
                 use_mirroring: str = False,
                 device: str = 'GPU', 
                 model_folder: str = None, 
                 use_folds: Tuple = None, 
                 checkpoint_name: str = None,
                 save_probabilities: bool = False) -> None:
        self.use_mirroring = use_mirroring
        self.device = device

        self.model_folder = model_folder
        self.use_folds = use_folds
        self.checkpoint_name = checkpoint_name

        self.save_probabilities = save_probabilities

        # Instantiate the nnUNetPredictor
        self.predictor = nnUNetPredictor(tile_step_size=0.5,
                                use_gaussian=True,
                                use_mirroring=self.use_mirroring,
                                perform_everything_on_device=True,
                                device=torch.device('cuda', 0) if self.device == 'GPU' else torch.device('cpu'),
                                verbose=False,
                                verbose_preprocessing=False,
                                allow_tqdm=True)
        
        # Initialize the network architecture, loads the checkpoint
        self.predictor.initialize_from_trained_model_folder(join(nnUNet_results, self.model_folder),
                                                            use_folds=self.use_folds,
                                                            checkpoint_name=self.checkpoint_name)
    
    def option_0001(self, input_dir: Union[str, List[List[str]]] = None, output_dir: Union[str, List[str]] = None):
        # Guarantee the output directory exists
        if not isinstance(input_dir, list):
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

        # Option 1 - Specified input and output folders
        self.predictor.predict_from_files(input_dir,
                                          output_dir,
                                          save_probabilities=self.save_probabilities, overwrite=False,
                                          num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                          folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

if __name__ == "__main__":
    input_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - nnUNet - TOTAL"
    output_dir = r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - Cranial Anatomy 1"

    # NIfTI_to_Neuroanatomy_Masks(input_dir, output_dir)

    use_mirroring = False
    device = 'GPU'
    model_folder = r"Dataset409_neuro_550subj\nnUNetTrainer_DASegOrd0__nnUNetPlans__3d_fullres_high"
    use_folds = (0,)
    checkpoint_name = 'checkpoint_final.pth'

    predictor = JGHPredictor(use_mirroring, device, model_folder, use_folds, checkpoint_name)

    input_dir = [[r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - Filepath Test\HK20240048470101\HK20240048470101.nii.gz"]]
    output_dir = [r"C:\Users\joshua.onichino\Dropbox\Head\Batch 5 - Filepath Test - Output\HK20240048470101\HK20240048470101.nii.gz"]

    predictor.option_0001(input_dir, output_dir)