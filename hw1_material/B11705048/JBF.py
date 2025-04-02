import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_s = sigma_s  # Spatial sigma
        self.sigma_r = sigma_r  # Range sigma
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        
        # 1. Pad the input image and guidance image to avoid boundary issues.
        padded_img = cv2.copyMakeBorder(
            img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        ).astype(np.int32)

        padded_guidance = cv2.copyMakeBorder(
            guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        ).astype(np.int32)

        # ---------------------------------------------------------------------
        # 2. Build two Look-Up Tables (LUTs):
        #    - table_G_s: for spatial Gaussian weights, indexed by [0..pad_w].
        #    - table_G_r: for range Gaussian weights, indexed by [0..255] (pixel differences).
        # ---------------------------------------------------------------------
        table_G_s = np.exp(-(np.arange(self.pad_w + 1) ** 2) / (2.0 * (self.sigma_s ** 2)))
        table_G_r = np.exp(-(np.arange(256) / 255.0) ** 2 / (2.0 * (self.sigma_r ** 2)))

        # Determine dimensions (for gray or color images).
        h, w = padded_img.shape[:2]
        if padded_img.ndim == 3:
            c = padded_img.shape[2]  # number of channels
        else:
            c = 1

        # Containers to accumulate weighted sums (result) and total weights (wgt_sum).
        result = np.zeros((h, w, c), dtype=np.float64)
        wgt_sum = np.zeros((h, w, c), dtype=np.float64)

        # ---------------------------------------------------------------------
        # 3. Double loop over the neighborhood offsets in both y and x:
        #    Use np.roll to shift the whole image and guidance accordingly.
        # ---------------------------------------------------------------------
        for dy in range(-self.pad_w, self.pad_w + 1):
            for dx in range(-self.pad_w, self.pad_w + 1):
                # 3-1. Compute the spatial weight (s_w) from the LUT using the offset distance.
                s_w = table_G_s[abs(dy)] * table_G_s[abs(dx)]

                # 3-2. Shift the guidance image by (dy, dx) and compute the difference with the original guidance.
                shifted_guidance = np.roll(padded_guidance, shift=(dy, dx), axis=(0, 1))
                diff = np.abs(shifted_guidance - padded_guidance)
                
                # For color guidance, combine the differences across channels before looking up table_G_r.
                if diff.ndim == 3 and diff.shape[2] == 3:
                    # Ensure differences are within [0, 255].
                    diff_clamped = np.clip(diff, 0, 255)
                    # Look up the range LUT for each channel -> shape: (H, W, 3).
                    r_w_rgb = table_G_r[diff_clamped]
                    # Multiply across channels to merge range weights.
                    r_w = np.prod(r_w_rgb, axis=2)
                else:
                    # For grayscale or single-channel guidance.
                    diff_clamped = np.clip(diff, 0, 255)
                    r_w = table_G_r[diff_clamped]

                # 3-3. Multiply spatial weight and range weight to form the total weight.
                #      s_w is a scalar, r_w is (H, W).
                t_w = s_w * r_w

                # 3-4. Shift the input image by (dy, dx) and accumulate into result and wgt_sum.
                shifted_img = np.roll(padded_img, shift=(dy, dx), axis=(0, 1))

                if c > 1:
                    # For color images with multiple channels.
                    result += shifted_img * t_w[..., None]
                    wgt_sum += t_w[..., None]
                else:
                    # For grayscale images, treat it as a single channel.
                    result[..., 0] += shifted_img[..., 0] * t_w
                    wgt_sum[..., 0] += t_w

        # ---------------------------------------------------------------------
        # 4. Normalize and remove the padding region to retrieve the final output size.
        # ---------------------------------------------------------------------
        # Avoid division by zero by adding a small epsilon.
        output = result / (wgt_sum + 1e-8)
        
        # Crop the padding (top, bottom, left, right).
        output = output[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w, :]

        # If it's a single-channel image, remove the trailing dimension.
        if c == 1:
            output = output[..., 0]

        # Clip to the range [0, 255] and convert back to uint8.
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
