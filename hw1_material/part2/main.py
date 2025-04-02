import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    # 1) Read the image in BGR format by default
    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read the image at {args.image_path}")

    # 2) Convert to RGB if you want to work in RGB space
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Also get a grayscale version (from BGR or from RGB - as long as you match the color space).
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    RGB_info = []

    with open(args.setting_path, 'r') as f:
        lines = f.readlines()
        sigma_line = lines[6].strip().split(',')
        sigma_s = int(float(sigma_line[1]))
        sigma_r = float(sigma_line[3])
        for i in range(1, 6):
            RGB_info.append(lines[i].rstrip('\n').strip().split(','))


    jbf = Joint_bilateral_filter(sigma_s, sigma_r)

    # Using the RGB image as guidance
    img_bf = jbf.joint_bilateral_filter(img_rgb, img_rgb)

    # Using the grayscale image as guidance
    img_jbf = jbf.joint_bilateral_filter(img_rgb, img_gray)

    cost_dict = {}
    cost_dict['COLOR_BGR2GRAY'] = np.sum(np.abs(img_bf.astype('int32') - img_jbf.astype('int32')))

    img_jbf_bgr = cv2.cvtColor(img_jbf, cv2.COLOR_RGB2BGR)

    if '1.png' in args.image_path:
        cv2.imwrite('./testdata/1_jbf_COLOR_BGR2GRAY.png', img_jbf_bgr)
        cv2.imwrite('./testdata/1_RGB2GRAY_grayscale.png', img_gray)
    else:
        cv2.imwrite('./testdata/2_jbf_COLOR_BGR2GRAY.png', img_jbf_bgr)
        cv2.imwrite('./testdata/2_RGB2GRAY_grayscale.png', img_gray)
        
    for i in RGB_info:
        r = i[0]
        g = i[1]
        b = i[2]
        
        # Convert to grayscale
        
        img_gray = img_rgb[:, :, 0] * float(r) + img_rgb[:, :, 1] * float(g) + img_rgb[:, :, 2] * float(b)
        img_jbf = jbf.joint_bilateral_filter(img_rgb, img_gray)
        
        cost_dict[f'{r},{g},{b}'] = np.sum(np.abs(img_bf.astype('int32') - img_jbf.astype('int32')))
        
        img_jbf = cv2.cvtColor(img_jbf,cv2.COLOR_BGR2RGB)
        
        if '1.png' in args.image_path:
            cv2.imwrite(f'./testdata/1_jbf_{r}_{g}_{b}.png', img_jbf)
            cv2.imwrite(f'./testdata/1_{r}_{g}_{b}_grayscale.png', img_gray)
        else:
            cv2.imwrite(f'./testdata/2_jbf_{r}_{g}_{b}.png', img_jbf)
            cv2.imwrite(f'./testdata/2_{r}_{g}_{b}_grayscale.png', img_gray)
            
        
        
        
    # Print the cost_dict properly
    for key in cost_dict:
        print(f"Cost for {key} is {cost_dict[key]}")
    
    # After all cost_dict entries have been populated:
    max_key = max(cost_dict, key=cost_dict.get)  
    min_key = min(cost_dict, key=cost_dict.get)  

    print(f"Highest cost is {cost_dict[max_key]}, from grayscale {max_key}")
    print(f"Lowest cost is {cost_dict[min_key]}, from grayscale {min_key}")


if __name__ == '__main__':
    main()
