import numpy as np # type: ignore
import cv2 # type: ignore

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_gaussian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, img):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        
        # First octave
        octave0_gaussians = [img]
        for idx in range(1, self.num_gaussian_images_per_octave):
            img_blurred = cv2.GaussianBlur(img, (0, 0), self.sigma**idx)
            octave0_gaussians.append(img_blurred)

        # Down-sample the last Gaussian image from the first octave
        downsampled_img = cv2.resize(
            octave0_gaussians[-1],
            (img.shape[1] // 2, img.shape[0] // 2),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Second octave
        octave1_gaussians = [downsampled_img]
        for idx in range(1, self.num_gaussian_images_per_octave):
            img_blurred = cv2.GaussianBlur(downsampled_img, (0, 0), self.sigma**idx)
            octave1_gaussians.append(img_blurred)

        # Combine both octaves
        all_gaussian_images = [octave0_gaussians, octave1_gaussians]

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        
        all_dog_images = []
        for oct_idx in range(self.num_octaves):
            # Collect DoG images for the current octave
            octave_dog_list = []
            cur_octave_gaussians = all_gaussian_images[oct_idx]

            # Generate DoGs by subtracting adjacent Gaussian images
            for i in range(self.num_DoG_images_per_octave):
                dog_diff = cv2.subtract(cur_octave_gaussians[i], cur_octave_gaussians[i + 1])
                octave_dog_list.append(dog_diff)

            # (Optional) Save normalized DoG images for visualization
            for i, dog_img in enumerate(octave_dog_list):
                dog_min, dog_max = dog_img.min(), dog_img.max()
                norm_dog = (dog_img - dog_min) * 255.0 / (dog_max - dog_min + 1e-8)
                norm_dog = norm_dog.astype(np.uint8)
                cv2.imwrite(f'./testdata/DoG{oct_idx + 1}-{i + 1}.png', norm_dog)

            all_dog_images.append(octave_dog_list)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for oct_idx in range(self.num_octaves):
            # Convert current octave's DoG images to a NumPy array: shape = (4, H, W)
            octave_dogs = np.array(all_dog_images[oct_idx])

            # Build a 3x3x3 neighborhood cube by rolling along axes=(2,1,0)
            #   z-axis: which DoG image, y-axis: image height, x-axis: image width
            neighborhood_cube = np.array([
                np.roll(octave_dogs, shift=(dx, dy, dz), axis=(2, 1, 0))
                for dz in range(-1, 2)
                for dy in range(-1, 2)
                for dx in range(-1, 2)
            ])

            # Threshold check (abs value >= threshold) AND local min/max check
            mask = (np.abs(octave_dogs) >= self.threshold) & (
                (octave_dogs == np.min(neighborhood_cube, axis=0)) |
                (octave_dogs == np.max(neighborhood_cube, axis=0))
            )

            # Only check DoG indices j=1, j=2 (skipping boundaries j=0 and j=3)
            for dog_idx in range(1, self.num_DoG_images_per_octave - 1):
                extremum_coords = np.argwhere(mask[dog_idx])
                for (y, x) in extremum_coords:
                    # For the second octave, scale coordinates by 2 to map back to the original image
                    if oct_idx == 1:
                        keypoints.append([y * 2, x * 2])
                    else:
                        keypoints.append([y, x])

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints), axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1], keypoints[:,0]))]

        return keypoints
