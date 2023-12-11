import scipy
import numpy as np

def dataload:
    dir_cover_pgm = r'F:\paksersht\Code Paksersht\V-Final Results_code\1-Feature Extraction\cover\0.4'
    dir_LSB_pgm = r'F:\paksersht\Code Paksersht\V-Final Results_code\1-Feature Extraction\LSB\0.4'
    dir_HUGO_pgm = r'F:\paksersht\Code Paksersht\V-Final Results_code\1-Feature Extraction\HUGO\0.4'
    dir_WOW_pgm = r'F:\paksersht\Code Paksersht\V-Final Results_code\1-Feature Extraction\WOW\0.4'
    dir_UNIWARD_pgm = r'F:\paksersht\Code Paksersht\V-Final Results_code\1-Feature Extraction\UNIWARD\0.4'
    dir_feature_ext_4 = r'F:\paksersht\Code Paksersht\V-Final Results_code\1-Feature Extraction\FeatureExtraction_0_4'

    mat_data = scipy.io.loadmat(dir_feature_ext_4)['Feature']
    data = np.array(mat_data)
    cover = data[0, :][0]
    LSB = data[1, :][0]
    WOW = data[2, :][0]
    HUGO = data[3, :][0]
    UNIWARD = data[4, :][0]
    data = np.concatenate((cover, LSB, WOW, HUGO, UNIWARD), axis=0)
    data = np.nan_to_num(data, nan=np.nanmean(data))
    label = np.zeros((len(data)))
    for i in range(5):
        for i2 in range(len(cover)):
            label[i * 1000 + i2] = i

    return data, label, dir_cover_pgm,dir_LSB_pgm, dir_UNIWARD_pgm,dir_HUGO_pgm,dir_WOW_pgm

