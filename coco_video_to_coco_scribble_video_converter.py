import argparse
import json
import numpy as np
import os
import copy
import math
import pycocotools.mask as mask_util
import multiprocessing

from detectron2.utils.file_io import PathManager
from davis_interactive_robot import InteractiveScribblesRobot

def get_parser():
    parser = argparse.ArgumentParser(description="COCO video to COCO scribble video format converter. \
        It saves the annotations for each video into the folder given by the parameter, then merged those into the final JSON. \
        The converter use a multiprocess pool for the conversation.")
    parser.add_argument(
        "--input-json-file",
        default="ytvis_2019/train.json",
        metavar="FILE",
        help="path to the input COCO video JSON file",
    )
    parser.add_argument(
        "--output-json-file-name",
        default="ytvis_2019_train_scribble.json", #"ytvis_2019/train_scribble.json",
        help="The name (path) of the output (merged) JSON file for the scribble format.",
    )
    parser.add_argument(
        "--saved-jsons-folder-name",
        default="ytvis_2019_train_scribble_annots",
        help="The name (path) of the folder, where the annotation JSON files for each video will be saved. Then those will be merged.",
    )
    parser.add_argument(
        "--output-json-file-prefix",
        default="ytvis_2019_train",
        help="The prefix of the names of the annotation JSON files which will be saved for each video. \
        e.g. with \"ytvis_2019_train\" the annotation JSON file names will be: ytvis_2019_train_vid_1.json, ytvis_2019_train_vid_2.json, etc.",
    )
    parser.add_argument(
        "--processes-number",
        type=int,
        default=8,
        help="Number of async worker processes. Consider the memory size of your device!",
    )
    return parser
    
class COCOVideoScribbleFormatConverter():
    """This class converts the COCO video format annotation into COCO video scribble format annotation.
    The process happens in five steps:
        1) read the input JSON file
        2) preprocess the annotations for each video to format, which the InteractiveScribblesRobot can use
        3) use the InteractiveScribblesRobot to obtain the scribble annotations
        4) save the annotation JSON files into the given folder for each video
        5) merge the saved annotation JSON files into the final JSON file
    """
    def __init__(self,input_json_file_name = 'ytvis_2019/train.json',output_json_file_name = 'ytvis_2019/train_scribble.json', saved_jsons_folder_name = 'ytvis_2019_train_scribble_annots', output_json_file_prefix = 'ytvis_2019_train'):
        self.input_json_file_name = input_json_file_name
        self.output_json_file_name = output_json_file_name
        self.saved_jsons_folder_name = saved_jsons_folder_name
        self.output_json_file_prefix = output_json_file_prefix
    """Create the criterion.
    Parameters:
        input_json_file_name (string): Path to the input COCO video JSON file
        output_json_file_name (string): The name (path) of the output (merged) JSON file for the scribble format.
        saved_jsons_folder_name (string): The name (path) of the folder, where the annotation JSON files for each video will be saved. Then those will be merged.
        output_json_file_prefix (string): The prefix of the names of the annotation JSON files which will be saved for each video.
    """
    
    def _is_continous_instance_ids_mask(self,mask):
        """
            Args:
                mask (ndarray): mask of a frame
            Returns:
                Bool:
                    wheter the instances IDs are following 
                    each others in the mask or not 
        """
        unique = np.unique(mask)
        len_unique = len(unique)
        return np.sum(unique) == (len_unique * (len_unique + 1)) / 2
    
    def _binary_mask_to_rle_np(self,binary_mask):
        """
            Args:
                binary_mask (ndarray): mask of a frame
            Returns:
                Dict:
                    Run Length Encoding of the annotation mask
        """
        rle = {"counts": [], "size": list(binary_mask.shape)}

        flattened_mask = binary_mask.ravel(order="F")
        diff_arr = np.diff(flattened_mask)
        nonzero_indices = np.where(diff_arr != 0)[0] + 1
        lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

        # note that the odd counts are always the numbers of zeros
        if flattened_mask[0] == 1:
            lengths = np.concatenate(([0], lengths))

        rle["counts"] = lengths.tolist()

        return rle
    
    def read_input_json_file(self):
        """
            Reads the input JSON file which is in COCO Video format.
        """
        with PathManager.open(self.input_json_file_name) as json_file:
            print('Loading from {} is started.'.format(self.input_json_file_name))
            self.json_in = json.load(json_file)
        self.vids_num = len(self.json_in['videos'])
        self.annots_num = len(self.json_in['annotations'])
        self.json_out = copy.deepcopy(self.json_in)
        print('Loaded {} videos.'.format(len(self.json_in['videos'])))
        print('Loaded {} annotations.'.format(len(self.json_in['annotations'])))
        self.json_out['annotations'] = []
    
    def _whole_mask_to_scribble_mask(self,vid_id,video_instance_mask,ins_id_annot_info_map,bg_annot_id):
            """
            Args:
                vid_id (int): ID of the video
                video_instance_mask (ndarray): mask of a video, in a format, which the InteractiveScribblesRobot can work with.
                ins_id_annot_info_map: a dictionary with at least the following elements:
                    id (int): ID of the annotation
                    category_id (int): ID of the object category
                    bboxes: bounding box annotations of the objects
                bg_annot_id (int): id of the background annotation for the video 
            Returns:
                list[dict]:
                    Scribble annotations for a video in COCO scribble format
            """
            
            # Cretaing dictionary about the annotations to work with DAVIS 
            segmentations_davis = {annot_info['id']:{'mask':[None]*video_instance_mask.shape[0],'category_id':annot_info['category_id'],'bboxes':annot_info['bboxes']} for annot_info in ins_id_annot_info_map.values()}
            segmentations_davis['bg'] = {'mask':[None]*video_instance_mask.shape[0]}
            
            robot = InteractiveScribblesRobot()
            print('Making scribble annotations for video with id {} ...'.format(vid_id))
            print('Video length: {} frames'.format(video_instance_mask.shape[0]))
            
            # Loop throught the frames of the mask
            for frame_idx in range(video_instance_mask.shape[0]):
                # using the davis-interactive tool to making scribble annotations
                if not self._is_continous_instance_ids_mask(video_instance_mask[frame_idx]):
                    # if instance ids are not following each other, then make them following  
                    ins_id_annot_id_map_curr_frame = {}
                    to_ins_id = 2
                    video_instance_mask_continous = np.ones_like(video_instance_mask)
                    for _ins_id in np.unique(video_instance_mask[frame_idx]):
                        if _ins_id == 1:
                            pass
                        elif _ins_id != to_ins_id:
                            ins_id_annot_id_map_curr_frame[to_ins_id] = ins_id_annot_info_map[_ins_id]['id']
                            video_instance_mask_continous[frame_idx][video_instance_mask[frame_idx] == _ins_id] = to_ins_id
                            to_ins_id += 1
                        else:
                            ins_id_annot_id_map_curr_frame[_ins_id] = ins_id_annot_info_map[_ins_id]['id']
                            video_instance_mask_continous[frame_idx][video_instance_mask[frame_idx] == _ins_id] = _ins_id
                            to_ins_id += 1
                    
                    robot_pred = robot.interact('test',
                        pred_masks=np.zeros_like(video_instance_mask, dtype=np.uint8),
                        gt_masks=video_instance_mask_continous,
                        nb_objects=np.unique(video_instance_mask_continous[frame_idx]).shape[0],
                        frame=frame_idx)
                else:
                    ins_id_annot_id_map_curr_frame = {key:value['id'] for key,value in ins_id_annot_info_map.items()}
                    robot_pred = robot.interact('test',
                        pred_masks=np.zeros_like(video_instance_mask, dtype=np.uint8),
                        gt_masks=video_instance_mask,
                        nb_objects=np.unique(video_instance_mask[frame_idx]).shape[0],
                        frame=frame_idx)
                
                # converting the output of the davis-interactive tool to binary masks
                prev_obj_id = 0
                for j in range(len(robot_pred['scribbles'][frame_idx])): # each predicted scribble from frame the given frames
                    obj_id = robot_pred['scribbles'][frame_idx][j]['object_id']
                    if obj_id != prev_obj_id:
                        out_mask = np.zeros((video_instance_mask.shape[1],video_instance_mask.shape[2]), dtype=np.uint8)
                    prev_obj_id = obj_id
                    for i in range(len(robot_pred['scribbles'][frame_idx][j]['path'])): # each point in the prediction
                        # convert them to the range of the mask
                        robot_pred['scribbles'][frame_idx][j]['path'][i][0] = math.floor(robot_pred['scribbles'][frame_idx][j]['path'][i][0] * video_instance_mask.shape[2])
                        robot_pred['scribbles'][frame_idx][j]['path'][i][1] = math.floor(robot_pred['scribbles'][frame_idx][j]['path'][i][1] * video_instance_mask.shape[1])
                        out_mask[robot_pred['scribbles'][frame_idx][j]['path'][i][1],robot_pred['scribbles'][frame_idx][j]['path'][i][0]] = 1
                    
                    if obj_id == 1:
                        segmentations_davis['bg']['mask'][frame_idx] = out_mask
                    else:
                        segmentations_davis[ins_id_annot_id_map_curr_frame[obj_id]]['mask'][frame_idx] = out_mask
                
                # converting the binary masks to RLE
                for key in segmentations_davis.keys():
                    if segmentations_davis[key]['mask'][frame_idx] is not None:
                        segmentations_davis[key]['mask'][frame_idx] = self._binary_mask_to_rle_np(np.asfortranarray(segmentations_davis[key]['mask'][frame_idx]))
                        
                print('Making scribble annotations for frame {} in video {} is ended.'.format(frame_idx+1,vid_id))
            
            annotations = []
            # filling up the json output with the annotations
            for annot_id,segmentations_info in segmentations_davis.items():
                if annot_id == 'bg':
                    annot_out = {
                        "id" : bg_annot_id, 
                        "video_id" : vid_id, 
                        "category_id" : None, 
                        "segmentations" : segmentations_info['mask'], 
                        "areas" : [None]*video_instance_mask.shape[0], 
                        "bboxes" : [[0.0, 0.0, float(video_instance_mask.shape[2]), float(video_instance_mask.shape[1])]]*video_instance_mask.shape[0], 
                        "iscrowd" : 0,
                    }
                else:
                    annot_out = {
                        "id" : annot_id, 
                        "video_id" : vid_id, 
                        "category_id" : segmentations_info['category_id'], 
                        "segmentations" : segmentations_info['mask'], 
                        "areas" : [None]*video_instance_mask.shape[0], 
                        "bboxes" : segmentations_info['bboxes'], 
                        "iscrowd" : 0,
                    }
                annotations.append(annot_out)
            
            print('-------------------------------------------------------------------------------------------')
            print('Scribble annotations has been made for video with id {}'.format(vid_id))
            if not os.path.exists(self.saved_jsons_folder_name):
                os.makedirs(self.saved_jsons_folder_name)
            output_json_file_name = './{}/{}_scribble_vid_{}.json'.format(self.saved_jsons_folder_name,self.output_json_file_prefix,vid_id)
            self._saving_scribble_annots_for_a_video(annotations, output_json_file_name = output_json_file_name)
            print('-------------------------------------------------------------------------------------------')
            return annotations
            
                
    def _preparing_mask_for_davis(self,annotations):
        """
        Does the conversation with multiple worker.
        Args:
            annotations: List of dictionaries of annotations for a video
        Returns:
            List of dictionaries of scribble annotations for a video
        """
        
        vid_id = annotations[0]['video_id']
        annot_ids_list = []
        
        ins_id = 1 # 1 is background, 0 is empty prediction input for davis interactive 
        video_instance_mask = None # it will be TxHxW
        ins_id_annot_info_map = {}
        for annot in annotations: # for each instance in the video
            
            print('Processing annotation with id {} ...'.format(annot['id']))
            annot_ids_list.append(annot['id'])
            # get the input annotations and convert those to binary masks
            instance_mask = None # it will be TxHxW
            
            for frame_idx,segm in enumerate(annot['segmentations']): # for each frame
                if segm is not None:
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                    decoded = mask_util.decode(segm)
                    if instance_mask is None:
                        instance_mask = np.zeros((len(annot['segmentations']),decoded.shape[0],decoded.shape[1]), dtype=np.uint8)
                    instance_mask[frame_idx] = decoded    
            if instance_mask is not None:
                ins_id += 1
                if video_instance_mask is None:
                    video_instance_mask = np.ones_like(instance_mask, dtype=np.uint8)
                video_instance_mask[instance_mask > 0] = ins_id
                ins_id_annot_info_map[ins_id] = {'id':annot['id'],'category_id':annot['category_id'],'bboxes':annot['bboxes']}

        return self._whole_mask_to_scribble_mask(vid_id,video_instance_mask,ins_id_annot_info_map,self.annots_num+min(annot_ids_list))
        
    def convert_dataset_to_scribble_annot_format_parallel(self,processes_num):
        """
        Does the conversation with multiple worker.
        Args:
            processes_num (int): Number of async worker processes.
        """
        
        print('number of processes: {}'.format(processes_num))
        self.vid_id_annot_idx_relations = {}
        for annot in self.json_in['annotations']:
            if annot['video_id'] in self.vid_id_annot_idx_relations.keys():
                self.vid_id_annot_idx_relations[annot['video_id']].append(annot['id']-1)
            else:
                self.vid_id_annot_idx_relations[annot['video_id']] = [annot['id']-1]

        # Create a pool with the given worker processes
        pool = multiprocessing.Pool(processes=processes_num)
        annotation_data = [[copy.deepcopy(self.json_in['annotations'][index]) for index in self.vid_id_annot_idx_relations[vid_id]] for vid_id in range(1,self.vids_num+1) if vid_id in self.vid_id_annot_idx_relations.keys()]
        annotation_data = annotation_data[:12]
        
        results = []
        
        # Submit tasks asynchronously for each video
        for vid_annot in annotation_data:
            result = pool.apply_async(self._preparing_mask_for_davis, (vid_annot,))
            results.append(result)
            
        # Wait for all tasks to complete and collect results
        final_results = [result.get() for result in results]
            
        pool.close()
        pool.join()
        
        self._merging_saved_annots(saved_jsons_folder_name = self.saved_jsons_folder_name, output_json_file_name = self.output_json_file_name)
        
    def _saving_scribble_annots_for_a_video(self, annotations, output_json_file_name):
        """
        Args:
            annotations: List of dictionaries of scribble annotations for a video
            output_json_file_name (string): Name of the output JSON file
        """
        print('Saving annotations to {}'.format(output_json_file_name))
        print('Saving {} annotations for {} videos.'.format(len(annotations),1))
        # Writing to sample.json
        with open(output_json_file_name, "w") as outfile:
            json.dump({'annotations':annotations}, outfile)
        print('Saving is done.')
        
    def _merging_saved_annots(self, saved_jsons_folder_name,output_json_file_name):
        """
        Args:
            saved_jsons_folder_name (string): Name of the folder, where the saved JSON file are for each video
            output_json_file_name (string): Name of the output JSON file
        """
        # iterate through all file
        for json_file in os.listdir(saved_jsons_folder_name):
            # Check whether file is in text format or not
            if json_file.endswith(".json"):
                json_file_name = saved_jsons_folder_name + '/' + json_file
                print('Loading annotations from {} is started...'.format(json_file_name))
                video_annot = json.load(open(json_file_name, "r"))
                self.json_out['annotations'].extend(video_annot['annotations'])
                print('Annotations sucessfully merged from {}.'.format(json_file_name))

        print('Saving annotations to {}'.format(output_json_file_name))
        print('Saving {} annotations for {} videos...'.format(len(self.json_out['annotations']),len(self.json_out['videos'])))
        with open(output_json_file_name, "w") as outfile:
            json.dump(self.json_out, outfile)
        print('Saving is done.')
        
            
if __name__ == "__main__":
    args = get_parser().parse_args()
    format_converter = COCOVideoScribbleFormatConverter(input_json_file_name = args.input_json_file, output_json_file_name = args.output_json_file_name, saved_jsons_folder_name = args.saved_jsons_folder_name, output_json_file_prefix = args.output_json_file_prefix)
    format_converter.read_input_json_file()
    format_converter.convert_dataset_to_scribble_annot_format_parallel(processes_num=8)