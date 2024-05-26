# Shree KRISHNAya Namaha
# Comprehensive module- takes input data, predicts Infilling Vectors and returns infilled frame n+1
# At test time alone, iterative infilling performed

import torch
from typing import Tuple
import torch.nn.functional as F


class Infiller(torch.nn.Module):
    def __init__(self, read_off_estimator, num_iterations: int = 3):
        super().__init__()
        self.read_off_estimator = read_off_estimator
        self.num_iterations = num_iterations
        return

    def forward(self, input_batch: dict):
        if self.training:
            read_off_output = self.read_off_estimator(input_batch)
            read_off_values = read_off_output['read_off_values']
            warped_frame4 = input_batch['warped_frame4']
            mask4_norm = input_batch['mask4'] / 255
            infilled_frame4, infilled_mask4 = self.bilinear_interpolation(warped_frame4, mask4_norm, read_off_values)
            result_dict = {}
            for key in read_off_output.keys():
                result_dict[key] = read_off_output[key]
            result_dict['predicted_frame4'] = infilled_frame4
        else:
            warped_frame4 = input_batch['warped_frame4']
            warped_depth4 = input_batch['warped_depth4']
            mask4 = input_batch['mask4']
            warped_infilling4 = input_batch['warped_infilling4']
            depth_threshold4 = input_batch['depth_threshold4']
            read_off_values_accumulator = torch.zeros_like(warped_infilling4)
            infilled_image, infilled_depth = None, None  # To avoid a warning
            for i in range(self.num_iterations):
                read_off_values, infilled_image, infilled_infilling_vector, infilled_depth, infilled_mask = \
                    self.iterative_infill(i, warped_frame4, mask4, warped_infilling4, warped_depth4, depth_threshold4)
                read_off_values_accumulator = self.accumulate_read_off_values(read_off_values_accumulator,
                                                                              read_off_values, mask4)

                # Set variables for next iteration
                warped_frame4 = infilled_image
                warped_infilling4 = infilled_infilling_vector
                warped_depth4 = infilled_depth
                mask4 = infilled_mask

            # A fail-safe to include only infilling in unknown regions
            warped_frame4 = input_batch['warped_frame4']
            warped_depth4 = input_batch['warped_depth4']
            mask4_norm = input_batch['mask4'] / 255
            infilled_image = mask4_norm * warped_frame4 + (1 - mask4_norm) * infilled_image
            infilled_depth = mask4_norm * warped_depth4 + (1 - mask4_norm) * infilled_depth
            infill_mask = mask4 * (1 - mask4_norm)

            result_dict = {
                'read_off_values': read_off_values_accumulator,
                'predicted_frame4': infilled_image,
                'infill_mask4': infill_mask,
                'infilled_depth4': infilled_depth,
            }
        return result_dict

    def iterative_infill(self, iter_num, warped_frame4, mask4, warped_infilling4, warped_depth4, depth_threshold4):
        # Ready data to pass to network
        read_off_input = {
            'warped_infilling4': warped_infilling4,
            'mask4': mask4,
            'warped_depth4': warped_depth4,
        }

        # Get network and STN predictions
        result_dict = self.read_off_estimator(read_off_input)
        read_off_values = result_dict['read_off_values']
        # The mask returned by interpolation is not used here, since the infilled_mask returned by
        # remove_foreground_infilling() is inclusive of this mask
        infilled_image, _ = self.bilinear_interpolation(warped_frame4, mask4, read_off_values)
        infilled_depth, _ = self.bilinear_interpolation(warped_depth4, mask4, read_off_values)

        if iter_num < self.num_iterations - 1:
            # Remove foreground infilling vectors and accumulate the read off values
            corrected_read_off_values, infilled_mask = self.remove_foreground_infilling(
                read_off_values, infilled_depth, depth_threshold4, mask4)
        else:
            # Last iteration: retain all infilling vectors. Increase magnitude of any infilling vectors
            # pointing to unknown regions
            _, infilled_mask = self.remove_foreground_infilling(
                read_off_values, infilled_depth, depth_threshold4, mask4)
            corrected_read_off_values = self.extend_infilling_vectors(read_off_values, mask4)
            # The mask obtained here will anyway be different from infilled_mask above, since this infilling vectors
            # contain foreground infilling vectors also
            infilled_image, _ = self.bilinear_interpolation(warped_frame4, mask4, corrected_read_off_values)
            infilled_depth, _ = self.bilinear_interpolation(warped_depth4, mask4, corrected_read_off_values)

        # Replace values in known pixels
        mask4_norm = mask4 / 255
        infilled_image = mask4_norm * warped_frame4 + (1 - mask4_norm) * infilled_image
        infilled_infilling_vector = mask4_norm * warped_infilling4 + (1 - mask4_norm) * corrected_read_off_values
        infilled_depth = mask4_norm * warped_depth4 + (1 - mask4_norm) * infilled_depth
        infilled_mask = mask4_norm * mask4 + (1 - mask4_norm) * infilled_mask
        return corrected_read_off_values, infilled_image, infilled_infilling_vector, infilled_depth, infilled_mask

    def accumulate_read_off_values(self, read_off_values_accumulator: torch.Tensor, read_off_values: torch.Tensor,
                                   mask4: torch.Tensor):
        b, _, h, w = mask4.shape
        grid = self.create_grid(b, h, w).to(read_off_values_accumulator)

        # Compute locations (known) where the accumulated read off values point to
        accumulated_read_off_locs = read_off_values_accumulator + grid

        # Compute locations (known) where current read off values point to (after adding previous read off
        # values)
        read_off_locs, read_off_mask = self.bilinear_interpolation(accumulated_read_off_locs, mask4,
                                                                   read_off_values)

        # Compute true read off values (i.e. after adding previous read off values)
        true_read_off = read_off_locs - grid

        # Wherever it can't read off or has already read in previous iteration, set those IV to 0
        true_read_off = true_read_off * read_off_mask * (1 - mask4 / 255)
        read_off_values_accumulator = read_off_values_accumulator + true_read_off
        return read_off_values_accumulator

    def bilinear_interpolation(self, frame2: torch.Tensor, mask2: torch.Tensor, flow12: torch.Tensor,
                               is_image: bool = False):
        """
        Using bilinear interpolation
        @param frame2: (b,c,h,w)
        @param mask2: (b,1,h,w): 1 for known, 0 for unknown
        @param flow12: (b,2,h,w)
        @param is_image: if true, output will be clamped to (-1,1) range
        :return: warped_frame1: (b,c,h,w)
                 mask1: (b,1,h,w): 1 for known and 0 for unknown
        """
        b, c, h, w = frame2.shape
        grid = self.create_grid(b, h, w).to(frame2)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1] - trans_pos_floor[:, 1])) * \
                         (1 - (trans_pos_offset[:, 0] - trans_pos_floor[:, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1] - trans_pos_offset[:, 1])) * \
                         (1 - (trans_pos_offset[:, 0] - trans_pos_floor[:, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1] - trans_pos_floor[:, 1])) * \
                         (1 - (trans_pos_ceil[:, 0] - trans_pos_offset[:, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1] - trans_pos_offset[:, 1])) * \
                         (1 - (trans_pos_ceil[:, 0] - trans_pos_offset[:, 0]))

        weight_nw_3d = prox_weight_nw[:, :, :, None]
        weight_sw_3d = prox_weight_sw[:, :, :, None]
        weight_ne_3d = prox_weight_ne[:, :, :, None]
        weight_se_3d = prox_weight_se[:, :, :, None]

        frame2_offset = F.pad(frame2, [1, 1, 1, 1])
        mask2_offset = F.pad(mask2, [1, 1, 1, 1])
        bi = torch.arange(b)[:, None, None]

        f2_nw = frame2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0]]
        f2_sw = frame2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]]
        f2_ne = frame2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]]
        f2_se = frame2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]]

        m2_nw = mask2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0]]
        m2_sw = mask2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]]
        m2_ne = mask2_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]]
        m2_se = mask2_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]]

        nr = weight_nw_3d * f2_nw * m2_nw + weight_sw_3d * f2_sw * m2_sw + \
             weight_ne_3d * f2_ne * m2_ne + weight_se_3d * f2_se * m2_se
        dr = weight_nw_3d * m2_nw + weight_sw_3d * m2_sw + weight_ne_3d * m2_ne + weight_se_3d * m2_se

        zero_tensor = torch.tensor(0, dtype=nr.dtype, device=nr.device)
        warped_frame1 = torch.where(dr > 0, nr / (dr + 1e-12), zero_tensor)
        mask1 = (dr > 0).float()

        # Convert to channel first
        warped_frame1 = warped_frame1.transpose(2, 3).transpose(1, 2)
        mask1 = mask1.transpose(2, 3).transpose(1, 2)

        if is_image:
            warped_frame1 = torch.clamp(warped_frame1, min=-1, max=1)
        return warped_frame1, mask1

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def remove_foreground_infilling(read_off_values: torch.Tensor, infilled_depth4: torch.Tensor,
                                    depth_threshold4: torch.Tensor, mask4: torch.Tensor):
        read_off_values = torch.clone(read_off_values)
        infill_mask = infilled_depth4 >= depth_threshold4
        mask4 = mask4 == 255
        # noinspection PyTypeChecker
        read_off_values[~torch.cat([infill_mask] * 2, dim=1)] = 0
        # noinspection PyUnresolvedReferences
        updated_mask = (mask4 | infill_mask).type(torch.FloatTensor).to(mask4.device) * 255
        return read_off_values, updated_mask

    def extend_infilling_vectors(self, infilling_vectors: torch.Tensor, mask2: torch.Tensor):
        """
        In a loop, do the following
        Search by increasing magnitudes by upto 100. Check if all unfilled pixels are filled now. If not, in the next
        iteration, use magnitudes from 101-200 and so on.
        """
        unfilled_mask = self.get_unfilled_mask(infilling_vectors, mask2)
        iv_mag_start, iter_num = 0, 0
        while unfilled_mask.sum() > 0:
            # Get locations of all unfilled pixels (infilling vectors pointing to unknown location). This is across
            # all batches. Get the corresponding infilling vectors and normalize them to unit magnitude.
            unfilled_indices = torch.nonzero(unfilled_mask).split(1, dim=1)
            unfilled_ivs = infilling_vectors[unfilled_indices[0], :, unfilled_indices[2], unfilled_indices[3]][:, 0, :]
            iv_norm = F.normalize(unfilled_ivs, p=2, dim=1)

            # Compute candidate infilling vectors by scaling their magnitudes
            magnitudes = torch.arange(iv_mag_start, iv_mag_start + 100)[None, :, None] + 1
            magnitudes = magnitudes.to(infilling_vectors)
            candidate_ivs = iv_norm[:, None, :].repeat([1, 100, 1]) * magnitudes

            # For the candidate infilling vectors, check if they're pointing to known region or not
            candidate_masks = self.compute_candidate_infilling_masks(mask2, unfilled_indices, candidate_ivs)

            # For each unfilled location, among the 100 candidates, get the first one (least magnitude) pointing to a
            # known pixel. iv_magnitudes_validity is 1 if such a candidate is found, else 0.
            iv_magnitudes, iv_magnitudes_validity = self.get_first_non_zero_index(candidate_masks, dim=1)

            # For each of the unfilled pixels, compute the extended infilling vector by scaling the magnitude of unit
            # infilling vector
            extended_ivs = (iv_magnitudes + 1) * iv_norm

            # Since for some of the unfilled locations, none of the 100 candidates point to a known region, select
            # only those for which valid extended infilling vectors are found.
            valid_extended_iv_indices = torch.nonzero(iv_magnitudes_validity)[:, 0]
            valid_unfilled_indices = [indices[valid_extended_iv_indices, 0] for indices in unfilled_indices]
            valid_extended_ivs = extended_ivs[valid_extended_iv_indices]

            # Create an array by populating the extended infilling vectors in their respective positions. In known
            # regions and unfilled regions where no valid extended infilling vectors are found, this array will have 0.
            # Also create the corresponding mask and then merge the existing infilling vectors with valid extended
            # infilling vectors
            extended_iv_array = torch.zeros_like(infilling_vectors)
            extended_iv_array[valid_unfilled_indices[0], :, valid_unfilled_indices[2], valid_unfilled_indices[3]] = \
                valid_extended_ivs
            extended_iv_mask = torch.zeros_like(unfilled_mask)
            extended_iv_mask[valid_unfilled_indices[0], :, valid_unfilled_indices[2], valid_unfilled_indices[3]] = 1
            infilling_vectors = (1 - extended_iv_mask) * infilling_vectors + extended_iv_mask * extended_iv_array

            # Compute the unfilled mask again, which has 1 for pixels which are unfilled yet
            unfilled_mask = self.get_unfilled_mask(infilling_vectors, mask2)
            iv_mag_start += 100
            iter_num += 1
            if iter_num >= 3:
                break
        return infilling_vectors

    def get_unfilled_mask(self, infilling_vectors: torch.Tensor, mask2: torch.Tensor):
        """
        @param infilling_vectors: (b,2,h,w)
        @param mask2: (b,1,h,w)
        @return unfilled_mask: (b,1,h,w): 1 for unfilled pixels and 0 and known/filled pixels. In borders, irrespective
        of whether it is known/filled/unfilled, it will be set to 0
        """
        b, c, h, w = infilling_vectors.shape
        grid = self.create_grid(b, h, w).to(infilling_vectors)
        read_off_src_locs = infilling_vectors + grid

        mask2 = mask2 / 255
        rosl_x = read_off_src_locs[:, 0]
        rosl_y = read_off_src_locs[:, 1]
        rosl_x_floor = torch.floor(rosl_x)
        rosl_x_ceil = torch.ceil(rosl_x)
        rosl_y_floor = torch.floor(rosl_y)
        rosl_y_ceil = torch.ceil(rosl_y)

        # If any of the 4 neighbors are known, then mark the pixel as filled.
        m1, m1v = self.sample_array_with_index_matrix(mask2, torch.stack([rosl_y_floor, rosl_x_floor], dim=1))
        m2, m2v = self.sample_array_with_index_matrix(mask2, torch.stack([rosl_y_ceil, rosl_x_floor], dim=1))
        m3, m3v = self.sample_array_with_index_matrix(mask2, torch.stack([rosl_y_floor, rosl_x_ceil], dim=1))
        m4, m4v = self.sample_array_with_index_matrix(mask2, torch.stack([rosl_y_ceil, rosl_x_ceil], dim=1))
        m1 = m1 * m1v
        m2 = m2 * m2v
        m3 = m3 * m3v
        m4 = m4 * m4v
        overall_mask = torch.clamp(m1 + m2 + m3 + m4, min=0, max=1)
        unfilled_mask = 1 - overall_mask
        return unfilled_mask

    def compute_candidate_infilling_masks(self, mask2: torch.Tensor, unfilled_indices: torch.Tensor,
                                          candidate_ivs: torch.Tensor):
        """
        @param mask2: (b,1,h,w)
        @param unfilled_indices: Tuple[(N,1), (N,1), (N,1), (N,1)]
        @param candidate_ivs: (N,100,2)
        """
        ib = unfilled_indices[0][:, 0]
        iy = unfilled_indices[2][:, 0][:, None] + candidate_ivs[:, :, 1]
        ix = unfilled_indices[3][:, 0][:, None] + candidate_ivs[:, :, 0]

        iy_floor = torch.floor(iy)
        iy_ceil = torch.ceil(iy)
        ix_floor = torch.floor(ix)
        ix_ceil = torch.ceil(ix)

        # If any of the 4 neighbors are known, then mark the pixel as filled.
        m1, m1v = self.sample_array_with_index_list(mask2, (ib, iy_floor, ix_floor))
        m2, m2v = self.sample_array_with_index_list(mask2, (ib, iy_ceil, ix_floor))
        m3, m3v = self.sample_array_with_index_list(mask2, (ib, iy_floor, ix_ceil))
        m4, m4v = self.sample_array_with_index_list(mask2, (ib, iy_ceil, ix_ceil))
        m1 = m1[:, :, 0] * m1v
        m2 = m2[:, :, 0] * m2v
        m3 = m3[:, :, 0] * m3v
        m4 = m4[:, :, 0] * m4v
        overall_mask = torch.clamp(m1 + m2 + m3 + m4, min=0, max=1)
        return overall_mask

    @staticmethod
    def sample_array_with_index_matrix(array: torch.Tensor, sampling_indices: torch.Tensor):
        """
        @param array: (b, c, h, w)
        @param sampling_indices: (b, 2, h, w)
        """
        b, c, h, w = array.shape
        iy = sampling_indices[:, 0]
        ix = sampling_indices[:, 1]

        valid_locs = (iy >= 0) & (iy <= h - 1) & (ix >= 0) & (iy <= w - 1)
        valid_locs = valid_locs[:, None]
        iy = torch.clamp(iy, min=0, max=h - 1)
        ix = torch.clamp(ix, min=0, max=w - 1)
        bi = torch.arange(b, dtype=torch.long, device=array.device)
        sampled_array = (array[bi.long()[:, None, None], :, iy.long(), ix.long()]).transpose(2, 3).transpose(1, 2)
        return sampled_array, valid_locs

    @staticmethod
    def sample_array_with_index_list(array: torch.Tensor, sampling_indices: Tuple[torch.Tensor, torch.Tensor,
                                                                                  torch.Tensor]):
        """
        @param array: (b, c, h, w)
        @param sampling_indices: Tuple[(N, 1), (N, 100), (N, 100)]
        """
        b, c, h, w = array.shape
        ib, iy, ix = sampling_indices
        valid_locs = (iy >= 0) & (iy <= h - 1) & (ix >= 0) & (iy <= w - 1)
        iy = torch.clamp(iy, min=0, max=h - 1)
        ix = torch.clamp(ix, min=0, max=w - 1)
        sampled_array = (array[ib[:, None].long(), :, iy.long(), ix.long()])
        return sampled_array, valid_locs

    @staticmethod
    def get_first_non_zero_index(array: torch.Tensor, dim: int):
        """
        @param array: (N, 100)
        @param dim: 1
        """
        idx = torch.arange(array.shape[dim], 0, -1).to(array)
        tmp = array * idx
        non_zero_indices = torch.argmax(tmp, dim, keepdim=True)
        valid_indices = array.sum(dim) > 0
        return non_zero_indices, valid_indices
