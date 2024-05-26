# Shree KRISHNAYa Namaha
# Loads frame4, warped_frame4, mask4, warped_infilling4, warped_depth4, depth_threshold4.

from pathlib import Path
from typing import Optional

# import Imath
# import OpenEXR
import numpy
import skimage.io
import skimage.transform

import torch
import torch.utils.data as data


class OurDataLoader(data.Dataset):
    """
    Loads frames of resolution 320x240.
    """

    def __init__(self, data_dirpath, split_name, patch_size):
        super(OurDataLoader, self).__init__()
        self.split_name = split_name
        self.generated_root_dirpath = Path(data_dirpath) / f'PrecomputedPriors/{self.split_name}'
        self.input_root_dirpath = Path(data_dirpath) / f'RenderedData/{self.split_name}'

        self.video_names = []
        for video_path in sorted(self.generated_root_dirpath.iterdir()):
            for i in [0, 150]:
                for j in range(3):
                    frame3_num = (i + j + 2) * 25
                    self.video_names.append((video_path.stem, frame3_num))
        self.patch_size = patch_size
        return

    def __getitem__(self, index):
        video_name, frame3_num = self.video_names[index]

        if self.split_name == 'Train':
            data_dict = self.load_training_data(video_name, frame3_num)
        else:
            data_dict = self.load_testing_data(video_name, frame3_num)
        return data_dict

    def load_training_data(self, video_name: str, frame3_num: int, patch_start_pt = None):
        frame4_num = frame3_num + 25

        frame4_path = self.input_root_dirpath / f'{video_name}/render/photo/{frame4_num:04}.jpg'
        warped_frame4_path = self.generated_root_dirpath / f'{video_name}/PoseWarping/1step/warped_frames/{frame4_num:04}.png'
        warped_depth4_path = self.generated_root_dirpath / f'{video_name}/PoseWarping/1step/warped_depths/{frame4_num:04}.npy'
        mask4_path = self.generated_root_dirpath / f'{video_name}/PoseWarping/1step/masks/{frame4_num:04}.png'
        disocc_mask4_path = self.generated_root_dirpath / f'{video_name}/PoseWarping/1step/disocclusion_masks/{frame4_num:04}.png'
        warped_infilling4_path = self.generated_root_dirpath / f'{video_name}/InfillingVector/warped_iv/{frame4_num:04}.npy'

        frame4 = self.get_image(frame4_path, patch_start_pt)
        warped_frame4 = self.get_image(warped_frame4_path, patch_start_pt)
        warped_depth4 = self.get_depth(warped_depth4_path, patch_start_pt)
        mask4 = self.get_mask(mask4_path, patch_start_pt)
        disocc_mask4 = self.get_mask(disocc_mask4_path, patch_start_pt)
        warped_infilling4 = self.get_warped_infilling_vector(warped_infilling4_path, patch_start_pt)

        data_dict = {
            'frame4': frame4,
            'warped_frame4': warped_frame4,
            'warped_depth4': warped_depth4,
            'mask4': mask4,
            'disocclusion_mask4': disocc_mask4,
            'warped_infilling4': warped_infilling4,
        }
        return data_dict

    def load_testing_data(self, video_name: str, frame3_num: int, patch_start_pt = None):
        
        frame4_num = frame3_num + 25

        if isinstance(video_name, int):
            video_name = f'{video_name:05}'

        frame4_path = self.input_root_dirpath / f'{video_name}/render/photo/{frame4_num:04}.jpg'
        warped_frame4_path = self.generated_root_dirpath / f'{video_name}/PoseWarping/1step/warped_frames/{frame4_num:04}.png'
        warped_depth4_path = self.generated_root_dirpath / f'{video_name}/PoseWarping/1step/warped_depths/{frame4_num:04}.npy'
        mask4_path = self.generated_root_dirpath / f'{video_name}/PoseWarping/1step/masks/{frame4_num:04}.png'
        disocc_mask4_path = self.generated_root_dirpath / f'{video_name}/PoseWarping/1step/disocclusion_masks/{frame4_num:04}.png'
        warped_infilling4_path = self.generated_root_dirpath / f'{video_name}/InfillingVector/warped_iv/{frame4_num:04}.npy'

        frame4 = self.get_image(frame4_path, patch_start_pt)
        warped_frame4 = self.get_image(warped_frame4_path, patch_start_pt)
        warped_depth4 = self.get_depth(warped_depth4_path, patch_start_pt)
        mask4 = self.get_mask(mask4_path, patch_start_pt)
        disocc_mask4 = self.get_mask(disocc_mask4_path, patch_start_pt)
        warped_infilling4 = self.get_warped_infilling_vector(warped_infilling4_path, patch_start_pt)
        depth_threshold4 = self.get_depth_threshold(warped_depth4.numpy(), mask4.numpy())

        data_dict = {
            'frame4': frame4,
            'warped_frame4': warped_frame4,
            'warped_depth4': warped_depth4,
            'mask4': mask4,
            'disocclusion_mask4': disocc_mask4,
            'warped_infilling4': warped_infilling4,
            'depth_threshold4': depth_threshold4,
        }
        return data_dict

    def __len__(self):
        return len(self.video_names)

    def get_image(self, path: Path, patch_start_point: Optional[tuple]):
        image = skimage.io.imread(path.as_posix())
        if patch_start_point is not None:
            h, w = patch_start_point
            image = image[h:h + self.patch_size, w:w + self.patch_size]
        image = image.astype(numpy.float32) / 255 * 2 - 1
        image_cf = numpy.moveaxis(image, [0, 1, 2], [1, 2, 0])
        image_tr = torch.from_numpy(image_cf)
        return image_tr

    def get_mask(self, path: Path, patch_start_point: Optional[tuple]):
        mask = skimage.io.imread(path.as_posix())
        if patch_start_point is not None:
            h, w = patch_start_point
            mask = mask[h:h + self.patch_size, w:w + self.patch_size]
        mask_cf = mask[None].astype(numpy.float32)
        mask_tr = torch.from_numpy(mask_cf)
        return mask_tr

    def get_depth(self, path: Path, patch_start_point: Optional[tuple]):
        if path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        else:  
            raise RuntimeError(f'Unknown depth format: {path.suffix}')

        if patch_start_point is not None:
            h, w = patch_start_point
            depth = depth[h:h + self.patch_size, w:w + self.patch_size]
        depth_cf = depth[None].astype(numpy.float32)
        depth_tr = torch.from_numpy(depth_cf)
        return depth_tr

    def get_warped_infilling_vector(self, path: Path, patch_start_point: Optional[tuple]):
        warped_infilling4 = numpy.load(path.as_posix())
        if patch_start_point is not None:
            h, w = patch_start_point
            warped_infilling4 = warped_infilling4[h:h + self.patch_size, w:w + self.patch_size]
        warped_infilling4 = warped_infilling4 / 2
        wi4_cf = numpy.moveaxis(warped_infilling4, [0, 1, 2], [1, 2, 0]).astype(numpy.float32)
        wi4_tr = torch.from_numpy(wi4_cf)
        return wi4_tr
    
    def get_depth_threshold(self, warped_depth4: numpy.ndarray, mask4: numpy.ndarray, num_neighbors: int = 5):
        warped_depth4 = self.convert_cf_to_cl(warped_depth4)
        mask4 = self.convert_cf_to_cl(mask4) == 255
        threshold_map = numpy.zeros(shape=mask4.shape)
        missing_pixels = numpy.argwhere(~mask4)
        for mp in missing_pixels:
            candidates = []
            # Check if there is a left valid pixel
            known_pixels = numpy.argwhere(mask4[mp[0], :mp[1]])[:, 0]
            for i in range(num_neighbors):
                if len(known_pixels) > i:
                    candidate_location = (mp[0], known_pixels[-1 - i])
                    if mask4[candidate_location]:
                        candidates.append(candidate_location)

            # Check if there is a right valid pixel
            known_pixels = numpy.argwhere(mask4[mp[0], mp[1]:])[:, 0]
            for i in range(num_neighbors):
                if len(known_pixels) > i:
                    candidate_location = (mp[0], mp[1] + known_pixels[i])
                    if mask4[candidate_location]:
                        candidates.append(candidate_location)

            # Check if there is a top valid pixel
            known_pixels = numpy.argwhere(mask4[:mp[0], mp[1]])[:, 0]
            for i in range(num_neighbors):
                if len(known_pixels) > i:
                    candidate_location = (known_pixels[-1 - i], mp[1])
                    if mask4[candidate_location]:
                        candidates.append(candidate_location)

            # Check if there is a bottom valid pixel
            known_pixels = numpy.argwhere(mask4[mp[0]:, mp[1]])[:, 0]
            for i in range(num_neighbors):
                if len(known_pixels) > i:
                    candidate_location = (mp[0] + known_pixels[i], mp[1])
                    if mask4[candidate_location]:
                        candidates.append(candidate_location)

            if len(candidates) > 4:
                candidate_depths = [warped_depth4[cl] for cl in candidates]
                num_extremes = min(num_neighbors, len(candidates) // 4)
                sorted_depths = numpy.sort(candidate_depths)
                min_depth = numpy.mean(sorted_depths[:num_extremes])
                max_depth = numpy.mean(sorted_depths[-num_extremes:])
                threshold = (max_depth + min_depth) / 2
                threshold_map[mp[0], mp[1], mp[2]] = threshold
        cf_threshold_map = self.convert_cl_to_cf(threshold_map)
        threshold_map_tr = torch.from_numpy(cf_threshold_map)
        return threshold_map_tr

    @staticmethod
    def convert_cl_to_cf(cl_array):
        """
        Converts channel last arrays to channel first
        """
        cf_array = numpy.moveaxis(cl_array, [0, 1, 2], [1, 2, 0])
        return cf_array

    @staticmethod
    def convert_cf_to_cl(cf_array):
        """
        Converts channel first arrays to channel last
        """
        cl_array = numpy.moveaxis(cf_array, [0, 1, 2], [2, 0, 1])
        return cl_array

    def get_video_names(self):
        video_names = sorted([path.stem for path in self.generated_root_dirpath.iterdir()])
        return video_names

    def load_test_data(self, video_name: str, frame3_num: int, patch_start_pt = None):
        data_dict = self.load_testing_data(video_name, frame3_num, patch_start_pt)

        for key in data_dict.keys():
            data_dict[key] = data_dict[key][None]
        return data_dict
