from skimage import io
import sys, argparse, os, imageio
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.transform import warp
from skimage import transform
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.feature import match_descriptors, corner_peaks, corner_harris, plot_matches, BRIEF
from skimage.measure import ransac
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np

def start_ransac(img1, img2, brief, common_factor=0.25):

    #https://www.researchgate.net/publication/264197576_scikit-image_Image_processing_in_Python
    img1 = transform.rescale(img1, common_factor, multichannel=False)
    img2 = transform.rescale(img2, common_factor, multichannel=False)

    print(img1.shape)
    print(img2.shape)

    if brief:
        #BRIEF
        keypoints1 = corner_peaks(corner_harris(img1), min_distance=5)
        keypoints2 = corner_peaks(corner_harris(img2), min_distance=5)

        extractor = BRIEF()

        extractor.extract(img1, keypoints1)
        keypoints1 = keypoints1[extractor.mask]
        descriptors1 = extractor.descriptors

        extractor.extract(img2, keypoints2)
        keypoints2 = keypoints2[extractor.mask]
        descriptors2 = extractor.descriptors

        matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
    else:
        #ORB
        orb = ORB(n_keypoints=1000, fast_threshold=0.05)

        orb.detect_and_extract(img1)
        keypoints1 = orb.keypoints
        desciptors1 = orb.descriptors

        orb.detect_and_extract(img2)
        keypoints2 = orb.keypoints
        desciptors2 = orb.descriptors

        matches12 = match_descriptors(desciptors1, desciptors2, cross_check=True)

    
    src = keypoints2 [ matches12[:, 1]][:, ::-1]
    dst = keypoints1 [ matches12[:, 0]][:, ::-1]

    model_robust, inliers = \
        ransac((src, dst), transform.SimilarityTransform, min_samples=4, residual_threshold=2)

    #r, c = img2.shape[:2]
    
    #corners = np.array([[0, 0],
    #    [0, r],
    #    [c, 0],
    #[c,r]])

    #warped_corners = model_robust(corners)
    #all_corners = np.vstack((warped_corners, corners))

    #corner_min = np.min(all_corners, axis=0)
    #corner_max = np.max(all_corners, axis=0)

    #output_shape = (corner_max - corner_min)
    #output_shape = np.ceil(output_shape[::-1])

    #offset = transform.SimilarityTransform(translation=-corner_min)
    
    #Not really cool rescaling
    #offset_tmatrix =  np.copy(offset.params)
    #offset_tmatrix[0, 2] = offset_tmatrix[0, 2]/common_factor
    #offset_tmatrix[0, 2] = offset_tmatrix[0, 2]/rescale_trans
    #offset_tmatrix[1, 2] = offset_tmatrix[1, 2]/common_factor
    #offset_tmatrix[1, 2] = offset_tmatrix[1, 2]/rescale_trans

    model_robust_tmatrix =  np.copy(model_robust.params)
    model_robust_tmatrix[0, 2] = model_robust_tmatrix[0, 2]/common_factor
    #model_robust_tmatrix[0, 2] = model_robust_tmatrix[0, 2]/rescale_trans
    model_robust_tmatrix[1, 2] = model_robust_tmatrix[1, 2]/common_factor
    #model_robust_tmatrix[1, 2] = model_robust_tmatrix[1, 2]/rescale_trans

    #model_robust_offset_tmatrix = np.copy((model_robust+offset).params)
    #model_robust_offset_tmatrix[0, 2] = offset_tmatrix[0, 2] + model_robust_tmatrix[0, 2]
    #model_robust_offset_tmatrix[1, 2] = offset_tmatrix[1, 2] + model_robust_tmatrix[1, 2]

    #factor2 = 1.05
    #img3_ = warp(img3, np.linalg.inv(offset_tmatrix), output_shape=(img3.shape[0]*factor2, img3.shape[1]*factor2))
    #img4_ = warp(img4, np.linalg.inv(model_robust_offset_tmatrix), output_shape=(img3.shape[0]*factor2, img3.shape[1]*factor2))


    img1_ = img1#= warp(img1, offset.inverse, output_shape=output_shape, cval=-1)
    img2_ = warp(img2, model_robust.inverse, cval=-1)#(model_robust+offset).inverse, output_shape=output_shape, cval=-1)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    f_ax1 = fig.add_subplot(gs[0, :])
    plot_matches(f_ax1, img1, img2, keypoints1, keypoints2, matches12)
    f_ax1.axis('off')
    #f_ax1.set_title(filename1+" vs. "+filename2)
    f_ax2 = fig.add_subplot(gs[1, 0])
    f_ax2.imshow(img1)
    f_ax2.axis('off')
    f_ax2.set_title("img1")
    f_ax3 = fig.add_subplot(gs[1, 1])
    f_ax3.imshow(img1_)
    f_ax3.axis('off')
    f_ax3.set_title("img1_")
    #f_ax4 = fig.add_subplot(gs[1, 2])
    #f_ax4.imshow(img3_)
    #f_ax4.axis('off')
    #f_ax4.set_title("img3_")
    f_ax5 = fig.add_subplot(gs[2, 0])
    f_ax5.imshow(img2)
    f_ax5.axis('off')
    f_ax5.set_title("img2")
    f_ax6 = fig.add_subplot(gs[2, 1])
    f_ax6.imshow(img2_)
    f_ax6.axis('off')
    f_ax6.set_title("img2_")
    #f_ax7 = fig.add_subplot(gs[2, 2])
    #f_ax7.imshow(img4_)
    #f_ax7.axis('off')
    #f_ax7.set_title("img4_")
    plt.show()

    return model_robust_tmatrix
    '''

    def add_alpha(image, background=-1):
        rgb = rgb2gray(image)
        alpha = (image != background)
        return np.dstack((image, alpha))

    img1_alpha = add_alpha(img1_)
    img2_aplha = add_alpha(img2_)

    merged = (img1_alpha + img1_alpha)

    alpha = merged[..., 3]

    merged /= np.maximum(alpha, 1)[..., np.newaxis]
    '''

def rescale_transform_matrix(matrix, factor):
    matrix[0, 2] = matrix[0, 2]/factor
    matrix[1, 2] = matrix[1, 2]/factor
    return matrix

def difference(a1, a2):
    a1 = a1.flatten()
    a2 = a2.flatten()
    return np.sum(np.absolute(a1 - a2))

def getMostSimilar(images, path):
    dif_list = list() 
    for elem1 in images:
        img1 = rgb2gray(io.imread(os.path.join(path, elem1)))
        dif = 0
        for elem2 in images:
            img2 = rgb2gray(io.imread(os.path.join(path, elem2)))
            dif += difference(img1, img2)
        dif_list.append(dif)
    return images[dif_list.index(min(dif_list))]

def doMostSimilar(files, path, real_path, output_path, brief):
    mostSimilar = getMostSimilar(files, path)
    mostSimilarImg = io.imread(os.path.join(path, mostSimilar))
    imageio.imwrite(output_path+mostSimilar+'base_ransac.png', img_as_ubyte(mostSimilarImg))
    mostSimilarRealImg = io.imread(os.path.join(real_path, mostSimilar[:-17]))
    imageio.imwrite(output_path+mostSimilar+'base_ransac.png', img_as_ubyte(mostSimilarRealImg))
    out = open(os.path.join(output_path, "eval.txt"), "a")
    out.write("Base: "+mostSimilar+"\n")
    evalu = 0
    for f in files:
        if not f==mostSimilar:
            img = io.imread(os.path.join(path, f))
            real_img = io.imread(os.path.join(real_path, f[:-17]))
            print(f,mostSimilar)
            reg = start_ransac(rgb2gray(mostSimilarImg), rgb2gray(img), mostSimilarRealImg, real_img, brief)
            imageio.imwrite(output_path+f+'_ransac.png', img_as_ubyte(reg))
            out.write(f+"\t"+str(difference(rgb2gray(mostSimilarImg), rgb2gray(img)))+"\t"+str(difference(rgb2gray(mostSimilarImg), rgb2gray(reg)))+"\n")
    out.close()

def doPropagate(files, path, output_path):
    img1 = io.imread(os.path.join(path, files[0]))
    img2 = io.imread(os.path.join(path, files[1]))
    reg_first = start_ransac(rgb2gray(img1), rgb2gray(img2), img1, img2)
    imageio.imwrite(output_path+files[0], img_as_ubyte(img1))
    imageio.imwrite(output_path+files[1]+'_ransac.png', img_as_ubyte(reg_first))
    out = open(os.path.join(output_path, "eval.txt"), "a")
    out.write("Base: "+files[0]+"\n")
    evalu = 0
    for i in range(1,len(files)-1):
        filename = files[i]+'_ransac.png'
        reg_img = io.imread(os.path.join(output_path, filename))
        img = io.imread(os.path.join(path, files[i+1]))
        reg = start_ransac(rgb2gray(reg_img), rgb2gray(img), reg_img, img)
        imageio.imwrite(output_path+files[i+1]+'_ransac.png', img_as_ubyte(reg))
        out.write(files[i+1]+"\t"+str(difference(rgb2gray(reg_img), rgb2gray(img)))+"\t"+str(difference(rgb2gray(reg_img), rgb2gray(reg)))+"\n")
    out.close()


def main():
    path = '/usr/local/hdd/rita/DL/model_ZT113_150/'
    output_path = '/usr/local/hdd/rita/registration/ransac/brief/mostSimilar/bla/'
    real_path = '/usr/local/hdd/rita/registration/ransac/brief/mostSimilar/original_he/'
    masks = sorted([f for f in os.listdir(path) if f.endswith('_dl.png')])
    he = sorted([f for f in os.listdir(real_path) if f.endswith('.tif')])

    mostSimilar_mask = getMostSimilar(masks, path)
    mostSimilarImg = io.imread(os.path.join(path, mostSimilar_mask))
    mostSimilar = mostSimilar_mask.split(".small")[0]
    mostSimilarRealImg = io.imread(os.path.join(real_path, mostSimilar))
    if False:
        imageio.imwrite(output_path+mostSimilar+'_ransac.png', img_as_ubyte(mostSimilarRealImg))

        out = open(os.path.join(output_path, "eval.txt"), "a")
        out.write("Base: "+mostSimilar+"\n")

        mostSimilarImg = rgb2gray(mostSimilarImg)

        for i in range(len(masks)):
            if not masks[i] == mostSimilar_mask:
                mask_img = rgb2gray(io.imread(os.path.join(path, masks[i])))

                real_name = masks[i].split(".small")[0]
                real_img = io.imread(os.path.join(real_path, real_name))

                rescale_trans = np.amin([ real_img.shape[0]/mask_img.shape[0], real_img.shape[1]/mask_img.shape[1] ])   

                trans = start_ransac(img1=mostSimilarImg, img2=mask_img, brief=True, common_factor=0.25)
                rescaled_transform = rescale_transform_matrix(trans, rescale_trans)

                reg = warp(real_img, np.linalg.inv(rescaled_transform))
                imageio.imwrite(output_path+real_name+'_ransac.png', img_as_ubyte(reg))
                
                out.write(real_name+"\t"+str(difference(mostSimilarImg, mask_img))+"\t"+str(difference(mostSimilarImg, rgb2gray(warp(mask_img, np.linalg.inv(trans)))))+"\n")

        out.close()
    chan_output_path = '/usr/local/hdd/rita/registration/ransac/brief/mostSimilar/channels_reg/'
    chan_path = '/usr/local/hdd/rita/registration/ransac/brief/mostSimilar/channels/'
    light = sorted([f for f in os.listdir(chan_path) if f.endswith('_light.jpg')])
    he_reg = sorted([f for f in os.listdir(output_path) if f.endswith('_ransac.png')])

    cha0 = sorted([f for f in os.listdir(chan_path) if f.endswith('_ch00.tif')])
    cha2 = sorted([f for f in os.listdir(chan_path) if f.endswith('_ch02.tif')])
    cha3 = sorted([f for f in os.listdir(chan_path) if f.endswith('_ch03.tif')])
    cha4 = sorted([f for f in os.listdir(chan_path) if f.endswith('_ch04.tif')])
    cha5 = sorted([f for f in os.listdir(chan_path) if f.endswith('_ch05.tif')])

    for i in range(len(he_reg)):
        light_img = io.imread(os.path.join(chan_path, light[i]))
        he_img =  io.imread(os.path.join(output_path,he_reg[i]))
        print(he_img.shape)
        if  light_img.shape[0]*light_img.shape[1] > he_img.shape[0]*he_img.shape[1]:
            first_rescale = np.amin([ light_img.shape[0]/he_img.shape[0], light_img.shape[1]/he_img.shape[1] ]) 
            light_resc = transform.rescale(light_img, 1/first_rescale, multichannel=False)
            print(light_img.shape)
            print(light_resc.shape)
            common_factor = 1 / np.amax([ light_resc.shape[0]/150, light_resc.shape[1]/150, he_img.shape[0]/150, he_img.shape[1]/150])
            trans = start_ransac(img1=rgb2gray(he_img), img2=rgb2gray(light_resc), brief=False, common_factor=common_factor)

            rescaled_transform = rescale_transform_matrix(trans, 1/first_rescale)
            reg = warp(light_img, np.linalg.inv(rescaled_transform))

            ch0 = io.imread(os.path.join(chan_path, cha0[i]))
            ch2 = io.imread(os.path.join(chan_path, cha2[i]))
            ch3 = io.imread(os.path.join(chan_path, cha3[i]))
            ch4 = io.imread(os.path.join(chan_path, cha4[i]))
            ch5 = io.imread(os.path.join(chan_path, cha5[i]))

            reg_ch0 = warp(ch0, np.linalg.inv(rescaled_transform))
            reg_ch2 = warp(ch2, np.linalg.inv(rescaled_transform))
            reg_ch3 = warp(ch3, np.linalg.inv(rescaled_transform))
            reg_ch4 = warp(ch4, np.linalg.inv(rescaled_transform))
            reg_ch5 = warp(ch5, np.linalg.inv(rescaled_transform))

            imageio.imwrite(chan_output_path+cha0[i]+'_ransac.png', img_as_ubyte(reg_ch0))
            imageio.imwrite(chan_output_path+cha2[i]+'_ransac.png', img_as_ubyte(reg_ch2))
            imageio.imwrite(chan_output_path+cha3[i]+'_ransac.png', img_as_ubyte(reg_ch3))
            imageio.imwrite(chan_output_path+cha4[i]+'_ransac.png', img_as_ubyte(reg_ch4))
            imageio.imwrite(chan_output_path+cha5[i]+'_ransac.png', img_as_ubyte(reg_ch5))

            imageio.imwrite(chan_output_path+light[i]+'_ransac.png', img_as_ubyte(reg))
        else:
            first_rescale = np.amin([ he_img.shape[0]/light_img.shape[0], he_img.shape[1]/light_img.shape[1] ]) 
            he_resc = transform.rescale(he_img, 1/first_rescale, multichannel=False)

            common_factor = 1 / np.amax([ light_img.shape[0]/150, light_img.shape[1]/150, he_resc.shape[0]/150, he_resc.shape[1]/150])
            trans = start_ransac(img1=rgb2gray(he_resc), img2=rgb2gray(light_img), brief=False, common_factor=common_factor)

            reg = warp(light_img, np.linalg.inv(trans))
            imageio.imwrite(chan_output_path+light[i]+'_ransac.png', img_as_ubyte(reg))





    #doMostSimilar(files, path, real_path, output_path, True)

    #light_path = '/usr/local/hdd/rita/registration/ransac/brief/mostSimilar/original_lights/'
    #light_files = sorted([f for f in os.listdir(light_path) if f.endswith('_light.jpg')])
    #for i in range(len(light_files)):


    #doPropagate(files, path, output_path)
    #img1 = rgb2gray(plt.imread('/usr/local/hdd/rita/registration/ransac/brief/mostSimilar/he/ZT13_6-1.tif.small.tif_dl.png_ransac.png'))
    #img2 = rgb2gray(plt.imread('/usr/local/hdd/rita/registration/ransac/brief/mostSimilar/6/AR_ZT13_6_1_light.jpg'))
    #img2 = transform.rescale(img2, 0.16, multichannel=False)
    #print(img1.shape)
    #print(img2.shape)
    #img3 = plt.imread('/usr/local/hdd/rita/registration/ransac/brief/mostSimilar/6/merged.jpg')
    #start_ransac(img1, img2, img1, img3)

if __name__ == "__main__":
    main()