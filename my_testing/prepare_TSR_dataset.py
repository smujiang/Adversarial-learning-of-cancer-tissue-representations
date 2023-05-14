# Import the relevant libraries from this module
from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
import multiprocessing
import os
import pandas as pd
import random

############### load case ID from list ######################################
# case_list = ["5045114_CR95-9505_B4_11-14-1995_HE.svs"]
############### load case ID from tsv ######################################
# train_case_number = 60
# case_metadata_fn = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/batch1_metadata.tsv"
# df = pd.read_csv(case_metadata_fn, sep="\t")
# case_list_batch_1 = [os.path.join("/infodev1/non-phi-data/junjiang/OvaryCancer/WSIs", i+".svs") for i in list(df["deidentified_id"])]
# wsi_fn_list = random.choices(case_list_batch_1, k=train_case_number)
############### load case ID from a format ######################################
wsi_fn_list = [os.path.join("/infodev1/non-phi-data/junjiang/OvaryCancer/WSIs", "OCMC-{:03d}.svs".format(i)) for i in
               range(1, 31)]
print(wsi_fn_list)

# Define some run parameters
num_processors = 10  # Number of processes that can be running at once

# Define a sample image that can be read by OpenSlide
output_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224"  # Define an output directory
log_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/Patches_out"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the parameters for Patch Extraction, including generating an thumbnail from which to traverse over to find
# tissue.
parameters = ExtractorParameters(output_dir,  # Where the patches should be extracted to
                                 save_format='.png',  # Can be '.jpg', '.png', or '.tfrecord'
                                 sample_cnt=-1,  # Limit the number of patches to extract (-1 == all patches)
                                 patch_size=448,  # Size of patches to extract (Height & Width)
                                 rescale_rate=128,  # Fold size to scale the thumbnail to (for faster processing)
                                 stride=448,
                                 patch_filter_by_area=0.5,  # Amount of tissue that should be present in a patch
                                 patch_rescale_to=224,
                                 with_anno=True,  # If true, you need to supply an additional XML file
                                 extract_layer=0,  # OpenSlide Level
                                 log_dir=log_dir
                                 )

# Choose a method for detecting tissue in thumbnail image
tissue_detector = TissueDetector("LAB_Threshold",  # Can be LAB_Threshold or GNB
                                 threshold=85,
                                 # Number from 1-255, anything less than this number means there is tissue
                                 training_files=None  # Training file for GNB-based detection
                                 )

# Create the extractor object
patch_extractor = PatchExtractor(tissue_detector,
                                 parameters,
                                 feature_map=None,  # See note below
                                 annotations=None  # Object of Annotation Class (see other note below)
                                 )

# Run the extraction process
# multiprocessing.set_start_method('spawn')
pool = multiprocessing.Pool(processes=num_processors)
pool.map(patch_extractor.extract, wsi_fn_list)
