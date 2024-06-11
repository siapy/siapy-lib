import numpy as np

import siapy.preparator.panel_pool_ftns as module
from siapy.utils.image_utils import merge_images_by_specter, save_image
from siapy.utils.utils import get_logger, init_ftn

logger = get_logger(name="preparator")


class Preparator:
    def __init__(self, config):
        self.cfg = config
        preparator_cfg = self.cfg.preparator

        self.slices_size_cam1 = preparator_cfg.image_slices_size_cam1
        self.slices_size_cam2 = preparator_cfg.image_slices_size_cam2
        self.labels_deliminator = preparator_cfg.labels_deliminator
        self.labels_deliminator_path = preparator_cfg.labels_path_deliminator
        self.percentage_bg = preparator_cfg.percentage_of_background
        self.ref_panel = preparator_cfg.reflectance_panel
        self.ref_panel_save = preparator_cfg.reflectance_panel_save
        self.panel_pool_ftn = preparator_cfg.panel_filter_function
        self.merge_images_by_specter = preparator_cfg.merge_images_by_specter

        self._post_init()

    def _post_init(self):
        # check if first object on an image represents reference panel
        if self.ref_panel is not None:
            if 0 < self.ref_panel < 1:
                logger.info(f"Reference reflactance used: {self.ref_panel}")
            else:
                logger.info("Reflectance panel not included.")
                self.ref_panel = False

        if self.slices_size_cam1 == -1:
            logger.info("Signatures will be created from whole objects.")
        elif isinstance(self.slices_size_cam1, int):
            logger.info(
                f"Signatures will be created from objects of size {[self.slices_size_cam1]*2} for camera1."
            )
            logger.info(
                f"Signatures will be created from objects of size {[self.slices_size_cam2]*2} for camera2."
            )
            if 0 < self.percentage_bg < 100:
                logger.info(
                    f"Background will be included in signatures with {self.percentage_bg}% of objects."
                )
            else:
                logger.error("Percentage of background must be in range (0, 100).")
        else:
            logger.warning("Parameter image_slices_size must be an integer!")

        # init filter function of reference panel
        self.panel_pool_ftn = init_ftn(module, self.panel_pool_ftn)

    def __run(self, images_segmented, slices_size):
        # first image is used to represent reference panel
        panel_correction = self._calculate_panel_correction(images_segmented[0])

        # remove reflectance image from processing
        if panel_correction is not None and not self.ref_panel_save:
            images_segmented.pop(0)

        # convert to reflectance image
        if panel_correction is not None:
            images_arr = [
                Preparator._convert_to_reflectance(image_segmented, panel_correction)
                for image_segmented in images_segmented
            ]

        # if no reference panel is used
        else:
            images_arr = [
                image_segmented.to_numpy() for image_segmented in images_segmented
            ]

        # whole converted image prepared for save
        if slices_size == -1:
            self._save_converted(images_segmented, images_arr)
        # converted image is further sliced and then saved
        else:
            images_slices_cam1 = [
                Preparator._blockfy(image_arr, slices_size, slices_size)
                for image_arr in images_arr
            ]
            images_slices_cam1 = [
                Preparator._remove_background_images(image_slices, self.percentage_bg)
                for image_slices in images_slices_cam1
            ]
            self._save_converted(images_segmented, images_slices_cam1)

    def run(self, images_segmented):
        if self.merge_images_by_specter and images_segmented.cam2 is not None:
            images_segmented = self._merge_images_spectrally(images_segmented)
            self.__run(images_segmented, self.slices_size_cam1)

        else:
            self.__run(images_segmented.cam1, self.slices_size_cam1)
            if images_segmented.cam2 is not None:
                self.__run(images_segmented.cam2, self.slices_size_cam2)

    def _merge_images_spectrally(self, images_segmented):
        images_segmented_out = []
        for image_cam1, image_cam2 in zip(images_segmented.cam1, images_segmented.cam2):
            data_file_name = f"images/merged/{image_cam1.filename}"
            image_merged = merge_images_by_specter(
                self.cfg.name, image_cam1, image_cam2, data_file_name=data_file_name
            )
            images_segmented_out.append(image_merged)
        return images_segmented_out

    def batch_images(self, images):
        images_out = list()
        images_temp = list()
        image_idx_hold = 0
        for image in images:
            image_idx = int(image.filename.split("__")[0].split("_")[0])
            if image_idx != image_idx_hold:
                image_idx_hold = image_idx
                if len(images_temp):
                    images_out.append(images_temp)
                images_temp = list()
                images_temp.append(image)
            else:
                images_temp.append(image)

        images_out.append(images_temp)
        return images_out

    def _save_converted(self, images_segmented, images_arr_prepared):
        # if image was sliced and slices need to be saved
        if isinstance(images_arr_prepared[0], list):
            for image_seg, images_list in zip(images_segmented, images_arr_prepared):
                for slice_idx, image_arr_prep in enumerate(images_list):
                    # split name of file
                    data_file_name = image_seg.filename.split("__")
                    # append slice index
                    data_file_name[0] += f"_{slice_idx}"
                    # merge back whole file name
                    data_file_name = "__".join(data_file_name)

                    data_file_name = f"images/converted/{data_file_name}"
                    metadata = image_seg.file.metadata
                    save_image(self.cfg, image_arr_prep, data_file_name, metadata)

        else:
            # if whole image is saved without slicing
            for image_seg, image_arr_prep in zip(images_segmented, images_arr_prepared):
                data_file_name = f"images/converted/{image_seg.filename}"
                metadata = image_seg.file.metadata
                save_image(self.cfg, image_arr_prep, data_file_name, metadata)

    def _calculate_panel_correction(self, image_segmented):
        # calculate panel radiance
        if self.ref_panel:
            image_segmented_arr = image_segmented.to_numpy()
            # remove nan values and get signatures
            image_mask = np.bitwise_not(
                np.bool_(np.isnan(image_segmented_arr).sum(axis=2))
            )
            panel_radiance = self.panel_pool_ftn(image_segmented_arr[image_mask])
            panel_reflectance = np.array(
                [self.ref_panel] * image_segmented_arr.shape[-1]
            )
            panel_correction = panel_reflectance / panel_radiance
        else:
            panel_correction = None
        return panel_correction

    @staticmethod
    def _remove_background_images(image_slices, percentage_bg):
        image_slices_out = []
        for image_slice in image_slices:
            # check where any of bands include nan values (axis=2) to get positions of background
            mask_nan = np.any(np.isnan(image_slice), axis=2)
            # calculate percentage of background
            percentage = np.sum(mask_nan) / mask_nan.size * 100
            # if percentage is lower than threshold
            if percentage < percentage_bg:
                # assign background if any nan values are present at any channel
                image_slice[mask_nan] = np.nan
                image_slices_out.append(image_slice)
        return image_slices_out

    @staticmethod
    def _convert_to_reflectance(image_segmented, panel_correction):
        return image_segmented.to_numpy() * panel_correction

    @staticmethod
    def _blockfy(image_arr, p, q):
        """
        Divides image into subarrays of size p-by-q
        p: block row size
        q: block column size
        """

        # caclulate how many whole block covers whole image
        bpr = (image_arr.shape[0] - 1) // p + 1  # blocks per row
        bpc = (image_arr.shape[1] - 1) // q + 1  # blocks per column

        # pad array with NaNs so it can be divided by p row-wise and by q column-wise
        A = np.nan * np.ones([p * bpr, q * bpc, image_arr.shape[2]])
        A[: image_arr.shape[0], : image_arr.shape[1], : image_arr.shape[2]] = image_arr

        image_slices = []
        previous_row = 0
        for row_block in range(bpc):
            previous_row = row_block * p
            previous_column = 0
            for column_block in range(bpr):
                previous_column = column_block * q
                block = A[
                    previous_row : previous_row + p,
                    previous_column : previous_column + q,
                ]

                if block.shape == (p, q, image_arr.shape[2]):
                    image_slices.append(block)

        return image_slices
