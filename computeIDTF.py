# extract IDT features

import numpy as np
import subprocess, os
import time
import sys, ffmpeg

"""
Wrapper library for the IDTF executable.
Implements methods to extract IDTFs.
Seperate methods for extracting IDTF and computing Fisher Vectors.

Assumes existance of tmpDir to store resized videos.


Example usage:

python computeIDTF.py video_list.txt output_directory

"""
#Path to the video repository
ucf101_path = "/home/xinqizhu/UCF101_frames"

# Improved Dense Trajectories binary
dtBin = '/home/xinqizhu/repo/iDT/release/DenseTrackStab'

# Temp directory to store resized videos
tmpDir = './tmp'


COMPUTE_FV = 'python ./computeFVstream.py'



def extract(videoName, outputBase, dataset):
    """
    Extracts the IDTFs and stores them in outputBase file.
    """
    if not dataset == "Something":
        videoName = videoName[:-4]
        if not os.path.exists(videoName):
            print '%s does not exist!' % videoName
            return False
        print(videoName)
        print(os.path.basename(videoName))
        resizedName = os.path.join(tmpDir, os.path.basename(videoName)+'/image_%05d.jpg')
        if not ffmpeg.resize(videoName, resizedName):
            resizedName = videoName+'/image_%05d.jpg'     # resize failed, just use the input video
    else:
        if not os.path.exists(videoName):
            print '%s does not exist!' % videoName
            return False
        print(videoName)
        print(os.path.basename(videoName))
        resizedName = os.path.join(tmpDir, os.path.basename(videoName)+'/%05d.jpg')
        if not ffmpeg.resize(videoName, resizedName):
            resizedName = videoName+'/%05d.jpg'     # resize failed, just use the input video

    subprocess.call('%s %s > %s' % (dtBin, resizedName, outputBase), shell=True)
    return True



def extractFV(videoName, outputBase,gmm_list, dataset):
    """
    Extracts the IDTFs, constructs a Fisher Vector, and saves the Fisher Vector at outputBase
    outputBase: the full path to the newly constructed fisher vector.
    gmm_list: file of the saved list of gmms
    """
    if not dataset == "Something":
        videoName = videoName[:-4]
        if not os.path.exists(videoName):
            print '%s does not exist!' % videoName
            return False
        resizedName = os.path.join(tmpDir, os.path.basename(videoName)+'/image_%05d.jpg')
        resized_vids = [filename for filename in os.listdir(tmpDir)]
        # resized_vids = [filename for filename in os.listdir(tmpDir) if filename.endswith('.avi')]
        if os.path.basename(videoName) not in resized_vids:
            if not ffmpeg.resize(videoName, resizedName):
                resizedName = videoName+'/image_%05d.jpg'     # resize failed, just use the input video
    else:
        if not os.path.exists(videoName):
            print '%s does not exist!' % videoName
            return False
        resizedName = os.path.join(tmpDir, os.path.basename(videoName)+'/%05d.jpg')
        resized_vids = [filename for filename in os.listdir(tmpDir)]
        # resized_vids = [filename for filename in os.listdir(tmpDir) if filename.endswith('.avi')]
        if os.path.basename(videoName) not in resized_vids:
            if not ffmpeg.resize(videoName, resizedName):
                resizedName = videoName+'/%05d.jpg'     # resize failed, just use the input video

    print videoName
    subprocess.call('%s %s | %s %s %s' % (dtBin, resizedName, COMPUTE_FV, outputBase, gmm_list), shell=True)
    return True


if __name__ == '__main__':
    #Useage: python computeIDTF.py video_list.txt output_directory
    videoList = sys.argv[1]
    outputBase = sys.argv[2]
    try:
        f = open(videoList, 'r')
        videos = f.readlines()
        f.close()
        videos = [video.rstrip() for video in videos]
        for i in range(0, len(videos)):
            outputName = os.path.join(outputBase, os.path.basename(videos[i])[:-4]+".features")
            videoLocation = os.path.join(ucf101_path,videos[i])
            print "generating IDTF for %s" % (videos[i],)
            extract(videoLocation, outputName)
            print "completed."
    except IOError:
        sys.exit(0)
